#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define N 500
#define MAX_LINE_LENGTH (N * 20)


static void load_csv_matrix(const char *path, double mat[N][N]);
static void dump_matrix_csv(const char *out_path, double mat[N][N]);
static double (*alloc_matrix_or_abort(void))[N];

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int my_rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double (*matA)[N] = NULL;
    double (*matB)[N] = NULL;
    double (*matC)[N] = NULL;

    const double t0 = MPI_Wtime();

    if (my_rank == 0) {
        matA = alloc_matrix_or_abort();
        matB = alloc_matrix_or_abort();
        matC = alloc_matrix_or_abort(); 

        load_csv_matrix("inputA_2000.csv", matA);
        load_csv_matrix("inputB_2000.csv", matB);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double acc = 0.0;
                for (int k = 0; k < N; ++k) {
                    acc += matA[i][k] * matB[k][j];
                }
                matC[i][j] = acc;
            }
        }

        dump_matrix_csv("output_seq.csv", matC);

        const double t1 = MPI_Wtime();
        printf("Tempo totale: %.6f secondi\n", (t1 - t0));

        free(matA);
        free(matB);
        free(matC);
    }

    MPI_Finalize();
    return 0;
}



static double (*alloc_matrix_or_abort(void))[N] {
    double (*m)[N] = (double (*)[N])malloc(sizeof(double[N][N]));
    if (!m) {
        fprintf(stderr, "Errore: malloc fallita per matrice %dx%d\n", N, N);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return m;
}

static void load_csv_matrix(const char *path, double mat[N][N]) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror(path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    char line[MAX_LINE_LENGTH];
    int row = 0;

    while (row < N && fgets(line, sizeof(line), fp)) {
        char *token = strtok(line, ",");
        int col = 0;

        while (col < N && token) {
            mat[row][col] = atof(token);
            token = strtok(NULL, ",");
            ++col;
        }
        ++row;
    }

    fclose(fp);
}

static void dump_matrix_csv(const char *out_path, double mat[N][N]) {
    FILE *out = fopen(out_path, "w");
    if (!out) {
        perror(out_path);
        exit(2);
    }

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            fprintf(out, "%6.1f", mat[r][c]);
            if (c < N - 1) fprintf(out, ",");
        }
        fprintf(out, "\n");
    }

    fclose(out);
    printf("Matrix %dx%d saved in %s\n", N, N, out_path);
}
