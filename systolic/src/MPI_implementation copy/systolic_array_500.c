#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// Global matrix size (square DIM x DIM)
#define DIM 500

// CSV I/O helpers
static void load_csv(const char *filename, double *matrix, int n);
static void save_csv(const char *filename, const double *matrix, int n);

int main(int argc, char *argv[]) {
    int rank = -1, world = 0;
    double t_start = 0.0, t_end = 0.0;

    // ---- Initialize MPI world ----
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    // ---- Validate arguments AFTER MPI so rank 0 can print nicely ----
    // Required: A.csv B.csv C_out.csv time_log.csv
    if (argc < 5) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <A.csv> <B.csv> <C_out.csv> <time_log.csv>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // ---- Start timing on root ----
    if (rank == 0) t_start = MPI_Wtime();

    // ---- Arrange processes in a p x p 2D grid ----
    int p = (int) sqrt((double)world);
    if (p * p != world) {
        if (rank == 0) fprintf(stderr, "Error: number of processes must be a perfect square.\n");
        MPI_Finalize();
        return 1;
    }
    if (DIM % p != 0) {
        if (rank == 0) fprintf(stderr, "Error: DIM (%d) must be divisible by p (%d).\n", DIM, p);
        MPI_Finalize();
        return 1;
    }

    const int block = DIM / p; // local block edge length

    // ---- Local tiles for A, B, and C ----
    double *A_loc = (double *) malloc((size_t)block * block * sizeof(double));
    double *B_loc = (double *) malloc((size_t)block * block * sizeof(double));
    double *C_loc = (double *) calloc((size_t)block * block, sizeof(double));

    // ---- Global matrices live only on rank 0 ----
    double *A = NULL, *B = NULL, *C = NULL;
    if (rank == 0) {
        A = (double *) malloc((size_t)DIM * DIM * sizeof(double));
        B = (double *) malloc((size_t)DIM * DIM * sizeof(double));
        C = (double *) malloc((size_t)DIM * DIM * sizeof(double));
        load_csv(argv[1], A, DIM);
        load_csv(argv[2], B, DIM);
    }

    // ---- Build counts and displacements for block scattering/gathering ----
    int *counts = (int *) malloc((size_t)world * sizeof(int));
    int *displs = (int *) malloc((size_t)world * sizeof(int));
    for (int r = 0, idx = 0; r < p; ++r) {
        for (int c = 0; c < p; ++c, ++idx) {
            counts[idx] = 1;                          // one sub-block per rank
            displs[idx] = r * DIM * block + c * block; // top-left element index of the sub-block
        }
    }

    // ---- Define a derived datatype representing a (block x block) tile with stride DIM ----
    MPI_Datatype tile_raw, tile;
    MPI_Type_vector(block, block, DIM, MPI_DOUBLE, &tile_raw);
    MPI_Type_create_resized(tile_raw, 0, sizeof(double), &tile);
    MPI_Type_commit(&tile);

    // ---- Scatter tiles of A and B to all ranks ----
    MPI_Scatterv(A, counts, displs, tile, A_loc, block * block, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, counts, displs, tile, B_loc, block * block, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ---- Create a 2D periodic Cartesian topology ----
    int dims[2] = { p, p };
    int periods[2] = { 1, 1 }; // wrap-around on both axes
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid);

    int coords[2];
    MPI_Cart_coords(grid, rank, 2, coords);
    const int my_row = coords[0];
    const int my_col = coords[1];

    int left, right, up, down;
    MPI_Cart_shift(grid, 1, -1, &right, &left); // column shifts (left/right neighbors)
    MPI_Cart_shift(grid, 0, -1, &down,  &up);   // row shifts (up/down neighbors)

    // ---- Initial Cannon alignment: shift A left by my_row, B up by my_col ----
    double *buf = (double *) malloc((size_t)block * block * sizeof(double));

    // Align A
    for (int s = 0; s < my_row; ++s) {
        MPI_Status st;
        if (((my_row + my_col) & 1) == 0) {
            MPI_Send(A_loc, block * block, MPI_DOUBLE, left,  100, grid);
            MPI_Recv(buf,   block * block, MPI_DOUBLE, right, 100, grid, &st);
        } else {
            MPI_Recv(buf,   block * block, MPI_DOUBLE, right, 100, grid, &st);
            MPI_Send(A_loc, block * block, MPI_DOUBLE, left,  100, grid);
        }
        memcpy(A_loc, buf, (size_t)block * block * sizeof(double));
    }

    // Align B
    for (int s = 0; s < my_col; ++s) {
        MPI_Status st;
        if (((my_row + my_col) & 1) == 0) {
            MPI_Send(B_loc, block * block, MPI_DOUBLE, up,   200, grid);
            MPI_Recv(buf,   block * block, MPI_DOUBLE, down, 200, grid, &st);
        } else {
            MPI_Recv(buf,   block * block, MPI_DOUBLE, down, 200, grid, &st);
            MPI_Send(B_loc, block * block, MPI_DOUBLE, up,   200, grid);
        }
        memcpy(B_loc, buf, (size_t)block * block * sizeof(double));
    }
    free(buf);

    // ---- Main Cannon loop: compute local product and rotate tiles ----
    for (int step = 0; step < p; ++step) {
        // Local GEMM: C_loc += A_loc * B_loc
        for (int i = 0; i < block; ++i) {
            for (int j = 0; j < block; ++j) {
                double acc = 0.0;
                for (int k = 0; k < block; ++k) {
                    acc += A_loc[i * block + k] * B_loc[k * block + j];
                }
                C_loc[i * block + j] += acc;
            }
        }

        // Rotate A left and B up by one block, avoiding deadlock with parity
        double *bufA = (double *) malloc((size_t)block * block * sizeof(double));
        double *bufB = (double *) malloc((size_t)block * block * sizeof(double));
        MPI_Status stA, stB;

        if (((my_row + my_col) & 1) == 0) {
            MPI_Send(A_loc, block * block, MPI_DOUBLE, left,  300, grid);
            MPI_Recv(bufA,  block * block, MPI_DOUBLE, right, 300, grid, &stA);
        } else {
            MPI_Recv(bufA,  block * block, MPI_DOUBLE, right, 300, grid, &stA);
            MPI_Send(A_loc, block * block, MPI_DOUBLE, left,  300, grid);
        }
        memcpy(A_loc, bufA, (size_t)block * block * sizeof(double));

        if (((my_row + my_col) & 1) == 0) {
            MPI_Send(B_loc, block * block, MPI_DOUBLE, up,   400, grid);
            MPI_Recv(bufB,  block * block, MPI_DOUBLE, down, 400, grid, &stB);
        } else {
            MPI_Recv(bufB,  block * block, MPI_DOUBLE, down, 400, grid, &stB);
            MPI_Send(B_loc, block * block, MPI_DOUBLE, up,   400, grid);
        }
        memcpy(B_loc, bufB, (size_t)block * block * sizeof(double));

        free(bufA);
        free(bufB);
    }

    // ---- Gather C tiles back to the root process ----
    MPI_Gatherv(C_loc, block * block, MPI_DOUBLE, C, counts, displs, tile, 0, MPI_COMM_WORLD);

    // ---- Persist outputs and timing on rank 0 ----
    if (rank == 0) {
        save_csv(argv[3], C, DIM);
        t_end = MPI_Wtime();
        double elapsed = t_end - t_start;

        FILE *tf = fopen(argv[4], "w");
        if (tf) {
            fprintf(tf, "%d,%.6f\n", world, elapsed);
            fclose(tf);
        } else {
            perror("Error opening time log file");
        }
    }

    // ---- Cleanup ----
    free(A_loc);
    free(B_loc);
    free(C_loc);
    free(counts);
    free(displs);

    if (rank == 0) {
        free(A);
        free(B);
        free(C);
    }

    MPI_Type_free(&tile);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return 0;
}

// Reads an n x n dense matrix from CSV into row-major buffer
static void load_csv(const char *filename, double *matrix, int n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening input matrix file");
        exit(1);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fscanf(fp, "%lf,", &matrix[i * n + j]) != 1) {
                fprintf(stderr, "Error reading element (%d,%d) from %s\n", i, j, filename);
                fclose(fp);
                exit(1);
            }
        }
    }
    fclose(fp);
}

// Writes an n x n dense matrix as CSV (row-major)
static void save_csv(const char *filename, const double *matrix, int n) {
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
