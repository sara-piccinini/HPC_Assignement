#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N 1000

void read_matrix_from_csv();
void write_matrix_to_csv();



int main(int argc, char *argv[]){
	
	int rank, size;
	double start_time, end_time, elapsed_time;
	
	if (argc < 5) {
		if (rank == 0) {
			printf("Missing input data\n", argv[0]);
		}
		MPI_Finalize();
		return 1;
	}
	MPI_Init(&argc, &argv);
	
	if (rank == 0) {
		start_time = MPI_Wtime();
	}
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	int p = (int)sqrt(size);	
	
	
	if (p*p!=size){
		if (rank==0){
			printf("Number of processes must be a perfect square.\n");
		}
		MPI_Finalize();
		return 1;
	}
	
	
	
	if (N%p!=0){
		if (rank==0){
			printf("Matrix size must be divisible by p = %d\n", p);
		}
		MPI_Finalize();
		return 1;
	}

	int block_size = N/p; 
	
	double *A_block = malloc(block_size * block_size * sizeof(double));	
	double *B_block = malloc(block_size * block_size * sizeof(double));
	double *C_block = calloc(block_size * block_size, sizeof(double));	
	
	double *A=NULL, *B=NULL, *C=NULL;
	
	
	if (rank==0){
		A = malloc(N * N * sizeof(double));
		B = malloc(N * N * sizeof(double));
		C = malloc(N * N * sizeof(double));
		read_matrix_from_csv(argv[1],A,N);
		read_matrix_from_csv(argv[2],B,N);
	}
	
	
	int *counts = malloc(size * sizeof(int));	
	int *displs = malloc(size * sizeof(int));	
	
	for (int i=0,idx=0; i<p; i++){
		for (int j=0; j<p; j++, idx++){
			counts[idx] = 1;
			displs[idx] = i * N * block_size + j * block_size;	
		}
	}
	
	
	MPI_Datatype blocktype1, blocktype2;
	MPI_Type_vector(block_size, block_size, N, MPI_DOUBLE, &blocktype1);	
	MPI_Type_create_resized(blocktype1, 0, sizeof(double), &blocktype2);
	MPI_Type_commit(&blocktype2);	
	
	
	MPI_Scatterv(A, counts, displs, blocktype2, A_block, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    	MPI_Scatterv(B, counts, displs, blocktype2, B_block, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	
	
	int dims[2] = {p, p};
    int periods[2] = {1, 1};
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
	
	int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];
	
	int left, right, up, down;
    MPI_Cart_shift(grid_comm, 1, -1, &right, &left); 
    MPI_Cart_shift(grid_comm, 0, -1, &down, &up);    
	
	
	
	double *tmp_block = malloc(block_size * block_size * sizeof(double));
	for (int i = 0; i < my_row; i++) {
    MPI_Status status;
    if ((my_col+my_row) % 2 == 0) {
        MPI_Send(A_block, block_size*block_size, MPI_DOUBLE, left, 0, grid_comm);
        MPI_Recv(tmp_block, block_size*block_size, MPI_DOUBLE, right, 0, grid_comm, &status);
		}
	else {
        MPI_Recv(tmp_block, block_size*block_size, MPI_DOUBLE, right, 0, grid_comm, &status);
        MPI_Send(A_block, block_size*block_size, MPI_DOUBLE, left, 0, grid_comm);
		}
    memcpy(A_block, tmp_block, block_size*block_size*sizeof(double));
	}
	for (int i = 0; i < my_col; i++) {
    MPI_Status status;
    if ((my_col+my_row) % 2 == 0) {
        MPI_Send(B_block, block_size*block_size, MPI_DOUBLE, up, 0, grid_comm);
        MPI_Recv(tmp_block, block_size*block_size, MPI_DOUBLE, down, 0, grid_comm, &status);
		} 
	else {
        MPI_Recv(tmp_block, block_size*block_size, MPI_DOUBLE, down, 0, grid_comm, &status);
        MPI_Send(B_block, block_size*block_size, MPI_DOUBLE, up, 0, grid_comm);
		}
    memcpy(B_block, tmp_block, block_size*block_size*sizeof(double));
	}
	free(tmp_block);
	
	
	
	for (int k = 0; k < p; k++) {
	        for (int i = 0; i < block_size; i++) {
	            for (int j = 0; j < block_size; j++) {
	                double sum = 0.0;
	                for (int l = 0; l < block_size; l++) {
	                    sum += A_block[i * block_size + l] * B_block[l * block_size + j];
	                }
	                C_block[i * block_size + j] += sum;
	            }
		}
	
	
        double *tmp_A = malloc(block_size * block_size * sizeof(double));
	double *tmp_B = malloc(block_size * block_size * sizeof(double));

		
		MPI_Status statusA;
		if ((my_col+my_row) % 2 == 0) {
			MPI_Send(A_block, block_size*block_size, MPI_DOUBLE, left, 0, grid_comm);
			MPI_Recv(tmp_A, block_size*block_size, MPI_DOUBLE, right, 0, grid_comm, &statusA);
			} 
		else {
			MPI_Recv(tmp_A, block_size*block_size, MPI_DOUBLE, right, 0, grid_comm, &statusA);
			MPI_Send(A_block, block_size*block_size, MPI_DOUBLE, left, 0, grid_comm);
			}
		memcpy(A_block, tmp_A, block_size*block_size*sizeof(double));
	
		MPI_Status statusB;
		if ((my_col+my_row) % 2 == 0) {
			MPI_Send(B_block, block_size*block_size, MPI_DOUBLE, up, 0, grid_comm);
			MPI_Recv(tmp_B, block_size*block_size, MPI_DOUBLE, down, 0, grid_comm, &statusB);
			} 
		else {
			MPI_Recv(tmp_B, block_size*block_size, MPI_DOUBLE, down, 0, grid_comm, &statusB);
			MPI_Send(B_block, block_size*block_size, MPI_DOUBLE, up, 0, grid_comm);
			}
		memcpy(B_block, tmp_B, block_size*block_size*sizeof(double));

		free(tmp_A);
		free(tmp_B);
    }

	
	MPI_Gatherv(C_block, block_size * block_size, MPI_DOUBLE, C, counts, displs, blocktype2, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		write_matrix_to_csv(argv[3], C, N);
	}
	
	
    free(A_block); free(B_block); free(C_block);
    free(counts); free(displs);
    MPI_Type_free(&blocktype2);
    MPI_Comm_free(&grid_comm);
	
	if (rank == 0) {
		end_time = MPI_Wtime();
		elapsed_time = end_time - start_time;

		FILE *fp = fopen(argv[4], "w"); 
		if (fp) {
			fprintf(fp, "%d,%.6f\n", size, elapsed_time);
			fclose(fp);
		} else {
			perror("Error in opening time file");
		}
	}
	
    MPI_Finalize();
    return 0;	
}


void read_matrix_from_csv(const char *filename, double *matrix, int n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error in opening input matrix file");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int result = fscanf(fp, "%lf,", &matrix[i*n + j]);
            if (result != 1) {
                fprintf(stderr, "Error in element reading [%d,%d]\n", i, j);
                fclose(fp);
                exit(1);
            }
        }
    }

    fclose(fp);
}

void write_matrix_to_csv(const char *filename, double *matrix, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error in opening output matrix file");
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp, "%.6f", matrix[i*n + j]);
            if (j < n-1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}