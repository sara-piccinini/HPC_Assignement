#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// Allocate a matrix (1D array for 2D)
double* allocate_matrix(int rows, int cols) {
    double* mat = (double*)malloc(rows * cols * sizeof(double));
    if (!mat) {
        fprintf(stderr, "Allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return mat;
}

void free_matrix(double* matrix) {
    free(matrix);
}

// Read CSV into matrix
double* read_matrix(const char* filename, int N) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error opening file %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double* mat = allocate_matrix(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            fscanf(f, "%lf,", &mat[i * N + j]);
        }
    }
    fclose(f);
    return mat;
}

void multiply_blocks(const double* A, const double* B, double* C, int block_size) {
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            double sum = 0.0;
            for (int k = 0; k < block_size; ++k) {
                sum += A[i * block_size + k] * B[k * block_size + j];
            }
            C[i * block_size + j] += sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) printf("Usage: %s <matrix_size>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    int P = (int)sqrt(size);
    if (P * P != size || N % P != 0) {
        if (rank == 0) fprintf(stderr, "Error: size must be a square number and N must be divisible by P\n");
        MPI_Finalize();
        return 1;
    }

    int block_size = N / P;

    MPI_Comm cart_comm;
    int dims[2] = {P, P};
    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int my_row = coords[0], my_col = coords[1];

    double* A = allocate_matrix(block_size, block_size);
    double* B = allocate_matrix(block_size, block_size);
    double* C = allocate_matrix(block_size, block_size);
    memset(C, 0, block_size * block_size * sizeof(double));

    double *A_full = NULL, *B_full = NULL;
    if (rank == 0) {
        A_full = read_matrix("input_matrices/A_2000.csv", N);
        B_full = read_matrix("input_matrices/B_2000.csv", N);
    }

    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < P; ++j) {
            int dest_coords[2] = {i, j}, dest_rank;
            MPI_Cart_rank(cart_comm, dest_coords, &dest_rank);
            if (rank == 0) {
                double* a_block = allocate_matrix(block_size, block_size);
                double* b_block = allocate_matrix(block_size, block_size);
                for (int r = 0; r < block_size; ++r)
                    for (int c = 0; c < block_size; ++c) {
                        a_block[r * block_size + c] = A_full[(i * block_size + r) * N + (j * block_size + c)];
                        b_block[r * block_size + c] = B_full[(i * block_size + r) * N + (j * block_size + c)];
                    }
                if (dest_rank == 0) {
                    memcpy(A, a_block, block_size * block_size * sizeof(double));
                    memcpy(B, b_block, block_size * block_size * sizeof(double));
                } else {
                    MPI_Send(a_block, block_size * block_size, MPI_DOUBLE, dest_rank, 0, cart_comm);
                    MPI_Send(b_block, block_size * block_size, MPI_DOUBLE, dest_rank, 1, cart_comm);
                }
                free_matrix(a_block);
                free_matrix(b_block);
            } else if (rank == dest_rank) {
                MPI_Recv(A, block_size * block_size, MPI_DOUBLE, 0, 0, cart_comm, MPI_STATUS_IGNORE);
                MPI_Recv(B, block_size * block_size, MPI_DOUBLE, 0, 1, cart_comm, MPI_STATUS_IGNORE);
            }
        }
    }

    int src_A, dst_A, src_B, dst_B;
    MPI_Cart_shift(cart_comm, 1, 1, &src_A, &dst_A);
    MPI_Cart_shift(cart_comm, 0, 1, &src_B, &dst_B);

    for (int i = 0; i < my_row; ++i)
        MPI_Sendrecv_replace(A, block_size * block_size, MPI_DOUBLE, dst_A, 0, src_A, 0, cart_comm, MPI_STATUS_IGNORE);

    for (int i = 0; i < my_col; ++i)
        MPI_Sendrecv_replace(B, block_size * block_size, MPI_DOUBLE, dst_B, 1, src_B, 1, cart_comm, MPI_STATUS_IGNORE);

    double start = MPI_Wtime();
    for (int step = 0; step < P; ++step) {
        multiply_blocks(A, B, C, block_size);
        MPI_Sendrecv_replace(A, block_size * block_size, MPI_DOUBLE, dst_A, 0, src_A, 0, cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(B, block_size * block_size, MPI_DOUBLE, dst_B, 1, src_B, 1, cart_comm, MPI_STATUS_IGNORE);
    }
    double end = MPI_Wtime();

    if (rank == 0) printf("Execution time: %f seconds\n", end - start);

    double* C_full = NULL;
    if (rank == 0) C_full = allocate_matrix(N, N);

    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < P; ++j) {
            int coords[2] = {i, j}, src_rank;
            MPI_Cart_rank(cart_comm, coords, &src_rank);
            if (rank == src_rank) {
                if (rank == 0) {
                    for (int r = 0; r < block_size; ++r)
                        for (int c = 0; c < block_size; ++c)
                            C_full[(i * block_size + r) * N + (j * block_size + c)] = C[r * block_size + c];
                } else {
                    MPI_Send(C, block_size * block_size, MPI_DOUBLE, 0, 2, cart_comm);
                }
            }
            if (rank == 0 && src_rank != 0) {
                double* block = allocate_matrix(block_size, block_size);
                MPI_Recv(block, block_size * block_size, MPI_DOUBLE, src_rank, 2, cart_comm, MPI_STATUS_IGNORE);
                for (int r = 0; r < block_size; ++r)
                    for (int c = 0; c < block_size; ++c)
                        C_full[(i * block_size + r) * N + (j * block_size + c)] = block[r * block_size + c];
                free_matrix(block);
            }
        }
    }

    if (rank == 0) {
        FILE* f = fopen("sara/test/C_out_2000_npX.csv", "w");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
                fprintf(f, "%lf,", C_full[i * N + j]);
            fprintf(f, "\n");
        }
        fclose(f);
    }

    free_matrix(A); free_matrix(B); free_matrix(C);
    if (rank == 0) { free_matrix(A_full); free_matrix(B_full); free_matrix(C_full); }
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}

