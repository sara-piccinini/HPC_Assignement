#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// Global matrix size (square N x N)
#define N 2000

// I/O helpers
void load_csv(const char *filename, double *matrix, int n);
void save_csv(const char *filename, const double *matrix, int n);

int main(int argc, char *argv[]) {
    // ---- Basic argument guard (no MPI needed to print this) ----
    if (argc < 5) {
        // argv[0] is the program name
        fprintf(stderr, "Usage: %s <A.csv> <B.csv> <C_out.csv> <time_log.csv>\n", argv[0]);
        return 1;
    }

    // ---- MPI bootstrap ----
    MPI_Init(&argc, &argv);

    int rank = -1, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Start timing only after MPI is up
    double t0 = 0.0;
    if (rank == 0) {
        t0 = MPI_Wtime();
    }

    // We arrange processes in a p x p Cartesian grid
    int p = (int) sqrt((double)world_size);
    if (p * p != world_size) {
        if (rank == 0) {
            fprintf(stderr, "Error: number of processes must be a perfect square.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // N must be divisible by p so each process gets a square block
    if (N % p != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: N (%d) must be divisible by p (%d).\n", N, p);
        }
        MPI_Finalize();
        return 1;
    }

    // Local block edge length
    const int block = N / p;

    // Local tiles
    double *A_loc = (double *) malloc((size_t)block * block * sizeof(double));
    double *B_loc = (double *) malloc((size_t)block * block * sizeof(double));
    double *C_loc = (double *) calloc((size_t)block * block, sizeof(double));

    // Global matrices (allocated only on root)
    double *A = NULL, *B = NULL, *C = NULL;
    if (rank == 0) {
        A = (double *) malloc((size_t)N * N * sizeof(double));
        B = (double *) malloc((size_t)N * N * sizeof(double));
        C = (double *) malloc((size_t)N * N * sizeof(double));
        load_csv(argv[1], A, N);
        load_csv(argv[2], B, N);
    }

    // ---- Build scatter metadata (counts/offsets in blocks) ----
    int *sendcounts = (int *) malloc((size_t)world_size * sizeof(int));
    int *displacements = (int *) malloc((size_t)world_size * sizeof(int));
    for (int i = 0, idx = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j, ++idx) {
            sendcounts[idx] = 1; // one block per destination rank
            displacements[idx] = i * N * block + j * block; // top-left index of block
        }
    }

    // ---- Define a derived datatype for a block (block x block, stride N) ----
    MPI_Datatype block_t_raw, block_t;
    MPI_Type_vector(block, block, N, MPI_DOUBLE, &block_t_raw);
    // Resize so blocks are tightly packed when scattering/gathering
    MPI_Type_create_resized(block_t_raw, 0, sizeof(double), &block_t);
    MPI_Type_commit(&block_t);

    // ---- Distribute tiles of A and B ----
    MPI_Scatterv(A, sendcounts, displacements, block_t, A_loc, block * block, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, sendcounts, displacements, block_t, B_loc, block * block, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ---- Create a 2D periodic Cartesian topology ----
    int dims[2] = { p, p };
    int periods[2] = { 1, 1 }; // wrap-around in both directions
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid);

    int coords[2];
    MPI_Cart_coords(grid, rank, 2, coords);
    const int my_row = coords[0];
    const int my_col = coords[1];

    int nbr_left, nbr_right, nbr_up, nbr_down;
    MPI_Cart_shift(grid, 1, -1, &nbr_right, &nbr_left); // shift in columns (left/right)
    MPI_Cart_shift(grid, 0, -1, &nbr_down, &nbr_up);    // shift in rows (up/down)

    // ---- Initial alignment (Cannon) ----
    double *buffer = (double *) malloc((size_t)block * block * sizeof(double));

    // Shift A left by my_row
    for (int s = 0; s < my_row; ++s) {
        MPI_Status st;
        if (((my_row + my_col) & 1) == 0) {
            MPI_Send(A_loc, block * block, MPI_DOUBLE, nbr_left, 0, grid);
            MPI_Recv(buffer, block * block, MPI_DOUBLE, nbr_right, 0, grid, &st);
        } else {
            MPI_Recv(buffer, block * block, MPI_DOUBLE, nbr_right, 0, grid, &st);
            MPI_Send(A_loc, block * block, MPI_DOUBLE, nbr_left, 0, grid);
        }
        memcpy(A_loc, buffer, (size_t)block * block * sizeof(double));
    }

    // Shift B up by my_col
    for (int s = 0; s < my_col; ++s) {
        MPI_Status st;
        if (((my_row + my_col) & 1) == 0) {
            MPI_Send(B_loc, block * block, MPI_DOUBLE, nbr_up, 0, grid);
            MPI_Recv(buffer, block * block, MPI_DOUBLE, nbr_down, 0, grid, &st);
        } else {
            MPI_Recv(buffer, block * block, MPI_DOUBLE, nbr_down, 0, grid, &st);
            MPI_Send(B_loc, block * block, MPI_DOUBLE, nbr_up, 0, grid);
        }
        memcpy(B_loc, buffer, (size_t)block * block * sizeof(double));
    }
    free(buffer);

    // ---- Main Cannon loop: compute + rotate tiles ----
    for (int step = 0; step < p; ++step) {
        // Local block multiply-add: C_loc += A_loc * B_loc
        for (int i = 0; i < block; ++i) {
            for (int j = 0; j < block; ++j) {
                double acc = 0.0;
                for (int k = 0; k < block; ++k) {
                    acc += A_loc[i * block + k] * B_loc[k * block + j];
                }
                C_loc[i * block + j] += acc;
            }
        }

        // Rotate A left and B up by one block (with deadlock avoidance)
        double *bufA = (double *) malloc((size_t)block * block * sizeof(double));
        double *bufB = (double *) malloc((size_t)block * block * sizeof(double));

        MPI_Status stA, stB;

        if (((my_row + my_col) & 1) == 0) {
            MPI_Send(A_loc, block * block, MPI_DOUBLE, nbr_left, 0, grid);
            MPI_Recv(bufA, block * block, MPI_DOUBLE, nbr_right, 0, grid, &stA);
        } else {
            MPI_Recv(bufA, block * block, MPI_DOUBLE, nbr_right, 0, grid, &stA);
            MPI_Send(A_loc, block * block, MPI_DOUBLE, nbr_left, 0, grid);
        }
        memcpy(A_loc, bufA, (size_t)block * block * sizeof(double));

        if (((my_row + my_col) & 1) == 0) {
            MPI_Send(B_loc, block * block, MPI_DOUBLE, nbr_up, 0, grid);
            MPI_Recv(bufB, block * block, MPI_DOUBLE, nbr_down, 0, grid, &stB);
        } else {
            MPI_Recv(bufB, block * block, MPI_DOUBLE, nbr_down, 0, grid, &stB);
            MPI_Send(B_loc, block * block, MPI_DOUBLE, nbr_up, 0, grid);
        }
        memcpy(B_loc, bufB, (size_t)block * block * sizeof(double));

        free(bufA);
        free(bufB);
    }

    // ---- Gather C blocks back to root ----
    MPI_Gatherv(C_loc, block * block, MPI_DOUBLE,
                C, sendcounts, displacements, block_t,
                0, MPI_COMM_WORLD);

    // ---- Persist outputs on rank 0 ----
    if (rank == 0) {
        save_csv(argv[3], C, N);

        double t1 = MPI_Wtime();
        double elapsed = t1 - t0;

        FILE *tf = fopen(argv[4], "w");
        if (tf) {
            // store (num_procs, elapsed_seconds)
            fprintf(tf, "%d,%.6f\n", world_size, elapsed);
            fclose(tf);
        } else {
            perror("Error opening time log file");
        }
    }

    // ---- Cleanup ----
    free(A_loc);
    free(B_loc);
    free(C_loc);
    free(sendcounts);
    free(displacements);

    if (rank == 0) {
        free(A);
        free(B);
        free(C);
    }

    MPI_Type_free(&block_t);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return 0;
}

// Reads a dense N x N matrix from a CSV file (row-major)
void load_csv(const char *filename, double *matrix, int n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening input matrix file");
        exit(1);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fscanf(fp, "%lf,", &matrix[i * n + j]) != 1) {
                fprintf(stderr, "Error parsing element (%d,%d) from %s\n", i, j, filename);
                fclose(fp);
                exit(1);
            }
        }
    }
    fclose(fp);
}

// Writes a dense N x N matrix to a CSV file (row-major)
void save_csv(const char *filename, const double *matrix, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error opening output matrix file");
        exit(1);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fprintf(fp, "%.6f", matrix[i * n + j]);
            if (j < n - 1) fputc(',', fp);
        }
        fputc('\n', fp);
    }
    fclose(fp);
}
