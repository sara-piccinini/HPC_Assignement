#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define N 2000   // Global matrix dimension

// Function prototypes
void load_matrix_csv(const char *filename, double *matrix, int n);
void save_matrix_csv(const char *filename, double *matrix, int n);

int main(int argc, char *argv[]) {
    int rank, size;
    double t_start, t_end;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Validate command line arguments
    if (argc < 5) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s matrixA.csv matrixB.csv result.csv time.csv\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Processes must form a perfect square grid
    int p = (int) sqrt(size);
    if (p * p != size) {
        if (rank == 0) {
            fprintf(stderr, "Error: number of processes (%d) must be a perfect square.\n", size);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Matrix dimension must be divisible by p
    if (N % p != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: matrix size %d not divisible by p=%d.\n", N, p);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int block_size = N / p;

    // Allocate local memory for sub-blocks
    double *A_block = malloc(block_size * block_size * sizeof(double));
    double *B_block = malloc(block_size * block_size * sizeof(double));
    double *C_block = calloc(block_size * block_size, sizeof(double));

    // Full matrices only stored on root process
    double *A = NULL, *B = NULL, *C = NULL;
    if (rank == 0) {
        A = malloc(N * N * sizeof(double));
        B = malloc(N * N * sizeof(double));
        C = malloc(N * N * sizeof(double));
        load_matrix_csv(argv[1], A, N);
        load_matrix_csv(argv[2], B, N);
        t_start = MPI_Wtime();
    }

    // Define counts and displacements for block distribution
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    for (int i = 0, idx = 0; i < p; i++) {
        for (int j = 0; j < p; j++, idx++) {
            counts[idx] = 1;
            displs[idx] = i * N * block_size + j * block_size;
        }
    }

    // Define MPI datatype for matrix block
    MPI_Datatype block_type, block_resized;
    MPI_Type_vector(block_size, block_size, N, MPI_DOUBLE, &block_type);
    MPI_Type_create_resized(block_type, 0, sizeof(double), &block_resized);
    MPI_Type_commit(&block_resized);

    // Scatter blocks to all processes
    MPI_Scatterv(A, counts, displs, block_resized,
                 A_block, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, counts, displs, block_resized,
                 B_block, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Create Cartesian topology
    int dims[2] = {p, p}, periods[2] = {1, 1};
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int my_row = coords[0], my_col = coords[1];

    int left, right, up, down;
    MPI_Cart_shift(grid_comm, 1, -1, &right, &left);
    MPI_Cart_shift(grid_comm, 0, -1, &down, &up);

    // Initial alignment (Cannon’s algorithm)
    double *tmp_block = malloc(block_size * block_size * sizeof(double));
    for (int i = 0; i < my_row; i++) {
        MPI_Status s;
        if ((my_row + my_col) % 2 == 0) {
            MPI_Send(A_block, block_size * block_size, MPI_DOUBLE, left, 0, grid_comm);
            MPI_Recv(tmp_block, block_size * block_size, MPI_DOUBLE, right, 0, grid_comm, &s);
        } else {
            MPI_Recv(tmp_block, block_size * block_size, MPI_DOUBLE, right, 0, grid_comm, &s);
            MPI_Send(A_block, block_size * block_size, MPI_DOUBLE, left, 0, grid_comm);
        }
        memcpy(A_block, tmp_block, block_size * block_size * sizeof(double));
    }
    for (int i = 0; i < my_col; i++) {
        MPI_Status s;
        if ((my_row + my_col) % 2 == 0) {
            MPI_Send(B_block, block_size * block_size, MPI_DOUBLE, up, 0, grid_comm);
            MPI_Recv(tmp_block, block_size * block_size, MPI_DOUBLE, down, 0, grid_comm, &s);
        } else {
            MPI_Recv(tmp_block, block_size * block_size, MPI_DOUBLE, down, 0, grid_comm, &s);
            MPI_Send(B_block, block_size * block_size, MPI_DOUBLE, up, 0, grid_comm);
        }
        memcpy(B_block, tmp_block, block_size * block_size * sizeof(double));
    }
    free(tmp_block);

    // Main Cannon’s algorithm loop
    for (int step = 0; step < p; step++) {
        // Local multiplication
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                double sum = 0.0;
                for (int k = 0; k < block_size; k++) {
                    sum += A_block[i * block_size + k] * B_block[k * block_size + j];
                }
                C_block[i * block_size + j] += sum;
            }
        }

        // Shift A left, B up
        double *tmp_A = malloc(block_size * block_size * sizeof(double));
        double *tmp_B = malloc(block_size * block_size * sizeof(double));

        MPI_Status sA, sB;
        if ((my_row + my_col) % 2 == 0) {
            MPI_Send(A_block, block_size * block_size, MPI_DOUBLE, left, 0, grid_comm);
            MPI_Recv(tmp_A, block_size * block_size, MPI_DOUBLE, right, 0, grid_comm, &sA);

            MPI_Send(B_block, block_size * block_size, MPI_DOUBLE, up, 0, grid_comm);
            MPI_Recv(tmp_B, block_size * block_size, MPI_DOUBLE, down, 0, grid_comm, &sB);
        } else {
            MPI_Recv(tmp_A, block_size * block_size, MPI_DOUBLE, right, 0, grid_comm, &sA);
            MPI_Send(A_block, block_size * block_size, MPI_DOUBLE, left, 0, grid_comm);

            MPI_Recv(tmp_B, block_size * block_size, MPI_DOUBLE, down, 0, grid_comm, &sB);
            MPI_Send(B_block, block_size * block_size, MPI_DOUBLE, up, 0, grid_comm);
        }

        memcpy(A_block, tmp_A, block_size * block_size * sizeof(double));
        memcpy(B_block, tmp_B, block_size * block_size * sizeof(double));
        free(tmp_A);
        free(tmp_B);
    }

    // Gather all partial results into root
    MPI_Gatherv(C_block, block_size * block_size, MPI_DOUBLE,
                C, counts, displs, block_resized, 0, MPI_COMM_WORLD);

    // Save final matrix and execution time
    if (rank == 0) {
        save_matrix_csv(argv[3], C, N);
        t_end = MPI_Wtime();
        FILE *tf = fopen(argv[4], "w");
        if (tf) {
            fprintf(tf, "%d,%.6f\n", size, t_end - t_start);
            fclose(tf);
        } else {
            perror("Error writing time file");
        }
    }

    // Cleanup
    free(A_block); free(B_block); free(C_block);
    free(counts); free(displs);
    if (rank == 0) { free(A); free(B); free(C); }
    MPI_Type_free(&block_resized);
    MPI_Comm_free(&grid_comm);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

// Load matrix from CSV
void load_matrix_csv(const char *filename, double *matrix, int n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("Input file error"); exit(1); }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fscanf(fp, "%lf,", &matrix[i * n + j]) != 1) {
                fprintf(stderr, "Error reading element (%d,%d)\n", i, j);
                fclose(fp);
                exit(1);
            }
        }
    }
    fclose(fp);
}

// Save matrix to CSV
void save_matrix_csv(const char *filename, double *matrix, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) { perror("Output file error"); exit(1); }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp, "%.6f", matrix[i * n + j]);
            if (j < n - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}
