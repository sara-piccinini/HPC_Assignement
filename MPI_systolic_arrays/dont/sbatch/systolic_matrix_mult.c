#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// func  that reads a matrix from CSV
void read_matrix(const char* filename, double* matrix, int N) {
    FILE* f = fopen(filename, "r");
    for (int i = 0; i < N * N; i++) {
        fscanf(f, "%lf,", &matrix[i]);
    }
    fclose(f);
}

// func that writes output matrix to CSV
void write_matrix(const char* filename, double* matrix, int N) {
    FILE* f = fopen(filename, "w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(f, "%.4f", matrix[i * N + j]);
            if (j < N - 1) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 4) {
        if (rank == 0)
            printf("Usage: mpirun -np <P> ./systolic_matrix_mult A.csv B.csv N\n");
        MPI_Finalize();
        return 1;
    }

    char* fileA = argv[1];
    char* fileB = argv[2];
    int N = atoi(argv[3]);

    // we allocate matrices on rank 0
    double *A = NULL, *B = NULL, *C = NULL;
    if (rank == 0) {
        A = malloc(N * N * sizeof(double));
        B = malloc(N * N * sizeof(double));
        C = malloc(N * N * sizeof(double));
        read_matrix(fileA, A, N);
        read_matrix(fileB, B, N);
    }

    //broadcast matrices A and B to all processes
    if (rank != 0) {
        A = malloc(N * N * sizeof(double));
        B = malloc(N * N * sizeof(double));
    }
    MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
if (rank == 0) {
    printf("DEBUG: First element A[0]: %f\n", A[0]);
    printf("DEBUG: First element B[0]: %f\n", B[0]);
}

   
    int rows_per_proc = N / world_size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == world_size - 1) ? N : start_row + rows_per_proc;

    double* C_part = calloc(N * (end_row - start_row), sizeof(double));

  
double t0 = MPI_Wtime();

for (int i = start_row; i < end_row; i++) {
    for (int j = 0; j < N; j++) {
        double sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C_part[(i - start_row) * N + j] = sum;
    }
}

// func to print time passed
double t1 = MPI_Wtime();

if (rank == 0) {
    printf("Elapsed time: %.6f seconds\n", t1 - t0);

    FILE *f = fopen("timing_results.txt", "a");
    if (f == NULL) {
        printf("ERROR: Could not open file for writing!\n");
    } else {
        fprintf(f, "size=%d, np=%d, time=%.6f\n", N, world_size, t1 - t0);
        fclose(f);
    }
}


printf("DEBUG: Rank %d finished computing.\n", rank);

    // Gather results at rank 0
    int rows_to_send = (end_row - start_row) * N;

MPI_Gather(C_part, rows_to_send, MPI_DOUBLE,
           C, rows_to_send, MPI_DOUBLE,
           0, MPI_COMM_WORLD);


    if (rank == 0) {
        write_matrix("C_out.csv", C, N);
        free(C);
    }

    free(A);
    free(B);
    free(C_part);
    MPI_Finalize();
    return 0;
}


