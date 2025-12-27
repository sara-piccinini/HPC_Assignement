#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define N 500
#define MAX_LINE_LENGTH (N * 20)


void read_csv(const char *fname, double matrix[N][N]);
void print_matrix(double matrix[N][N]);



int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Definisco i blocchi di A e B
    double (*C)[N] = NULL;
    double (*A)[N] = NULL;
    double (*B)[N] = NULL;

    double start_time = MPI_Wtime();

    if(rank == 0) {
        A = malloc(sizeof(double[N][N]));
        B = malloc(sizeof(double[N][N]));
        C = malloc(sizeof(double[N][N]));
        read_csv("inputA_2000.csv",  A);
        read_csv("inputB_2000.csv",  B);


        double C_verify[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C_verify[i][j] = 0;
                for (int k = 0; k < N; k++) {
                    C_verify[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        print_matrix(C_verify);

    }

    if(rank == 0) {
        double end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;
        printf("Tempo totale: %.6f secondi\n", elapsed_time);
        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();


    return 0;
}

void read_csv(const char *fname, double matrix[N][N]) {
    FILE *fp = fopen(fname, "r");
    if (!fp) {
        perror(fname);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    char line[MAX_LINE_LENGTH];
    int r = 0;
    while (fgets(line, sizeof(line), fp) && r < N) {
        char *tok = strtok(line, ",");
        int c = 0;
        while (tok && c < N) {
            matrix[r][c] = atof(tok);
            tok = strtok(NULL, ",");
            c++;
        }
        r++;
    }
    fclose(fp);
}

void print_matrix(double matrix[N][N]) {
    FILE* out = fopen("output_seq.csv", "w");
    if(out == NULL) {
        printf("Error opening the output file");
        exit(2);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(out, "%6.1f", matrix[i][j]);
            //printf("%6.1f ", matrix[i][j]);
            if (j < N - 1)
                fprintf(out, ",");
        }
        fprintf(out, "\n");
        //printf("\n");
    }
    fclose(out);
    printf("Matrix %dx%d saved in output_NxN_seq.csv\n", N, N);
}