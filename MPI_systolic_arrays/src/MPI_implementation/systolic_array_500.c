#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

//Definition of the size of the column/row of the matrix
#define N 500 


//Definition of the functions to read the matrices from the .csv files and write the output matrix
void read_matrix_from_csv();
void write_matrix_to_csv();



int main(int argc, char *argv[]){
	
	//Definition of rank, size, variables to compute the time
	int rank, size;
	double start_time, end_time, elapsed_time;
	
	//Check if there are all the argument as input (executable program name, matrix A file, matrix B file, outupt matrix file)
	//If there are not all the input arguments close process
	if (argc < 5) {
		if (rank == 0) {
			printf("Missing inputs\n", argv[0]);
		} 
		return 1;
	}
	
	//MPI initialization 
	MPI_Init(&argc, &argv);
	
	//Take initial time with rank = 0 process
	if (rank == 0) {
		start_time = MPI_Wtime();
	}
	
	//Set the rank of the processes in the MPI_COMM_WORLD
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	//Set the number of processes in the MPI_COMM_WORLD
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	//Definition of p as the square root of the number of processes
	int p = (int)sqrt(size);	
	
	//Check if the number of processes are a perfect square, if not finalize MPI and close process
	if (p*p!=size){
		if (rank==0){
			printf("Number of processes must be a perfect square.\n");
		}
		MPI_Finalize();
		return 1;
	}
	
	
	//Check if the matrix of size N is divisible for p, if not finalize MPI and close process. Done 
	//to check if the matrix of size NxN can be divided in p*p (size) processes
	if (N%p!=0){
		if (rank==0){
			printf("Matrix size must be divisible by p = %d\n", p);
		}
		MPI_Finalize();
		return 1;
	}
	
	//Definition of block_size (row and columns size for the PEs)
	int block_size = N/p; 
	
	//Definition in the dynamic memory of the size of the PEs
	double *A_block = malloc(block_size * block_size * sizeof(double));	
	double *B_block = malloc(block_size * block_size * sizeof(double));
	double *C_block = calloc(block_size * block_size, sizeof(double));	
	
	//Matrix A, B, C initialization
	double *A=NULL, *B=NULL, *C=NULL;
	
	//Allocate in the dynamic memory the matrices A, B and C, with their size definition.
	//Call for the functions to read from the .csv file the matrices
	if (rank==0){
		A = malloc(N * N * sizeof(double));
		B = malloc(N * N * sizeof(double));
		C = malloc(N * N * sizeof(double));
		read_matrix_from_csv(argv[1],A,N);
		read_matrix_from_csv(argv[2],B,N);
	}
	
	
	//Definition of the dynamic arrays. counts indicates the number of blocks I send/receive every process (always 1 in my case)
	//displs indicates the position (as offset) of the global matrices from which the block to send/receive is taken. Fundamental
	//to allow every PE to have its own block
	int *counts = malloc(size * sizeof(int));	
	int *displs = malloc(size * sizeof(int));	
	
	//Cicle to fill the counts and displs vectors. i flows with rows, j flows with columns. The grid is p*p, so 
	//both go from 0 to (p-1). idx is the index that goes from 0 to (size-1), indicates the position of the vector to fill
	for (int i=0,idx=0; i<p; i++){
		for (int j=0; j<p; j++, idx++){
			counts[idx] = 1;
			displs[idx] = i * N * block_size + j * block_size;	
		}
	}
	
	//Initialization of a new type of data in MPI
	MPI_Datatype blocktype1, blocktype2;
	
	//Definition of this new type of data as a square block of dimension block_size*block_size (PEs size).
	//N tells how many elements we have to skip to go from one row to the other. MPI_DOUBLE is the type of the basic elements
	//&blocktype1 saves the new data in this variable
	MPI_Type_vector(block_size, block_size, N, MPI_DOUBLE, &blocktype1);
		
	//Function to define a ne data type starting from one already existent (blocktype1). '0' is the initial displacemente
	//(like offset), sizeof(double) is the distance in memory between one element and the other when we use this data, like 
	//an array. &blocktype2 is where we save it
	MPI_Type_create_resized(blocktype1, 0, sizeof(double), &blocktype2);
	
	//Function to register this new type of data in the MPI system
	MPI_Type_commit(&blocktype2);	
	
	//Functions to send from process '0' to the other processes the blocks of the input matrices. 'A' and 'B' are the 
	//pointers of matrix A and B. 'counts' is the number of block every process receive, 'displs' from where in the global matrix
	//the block is sent. 'blocktype2' the type of block sent, 'A_block' and 'B_block' local pointers where the blocks are received
	//'block_size * block_size' the size, 'MPI_DOUBLE' type of basic element of blocktype2 (received), '0' rank of the process
	//with the global matrix and 'MPI_COMM_WORLD' communicator
	MPI_Scatterv(A, counts, displs, blocktype2, A_block, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, counts, displs, blocktype2, B_block, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	//CARTESIAN GRID INSTANTIATION
	
	//Definition of the grid dimension (2D), p columns and p rows
	int dims[2] = {p, p};
	
	//Definition of the periodicity in the cartesian grid. With '1' for both, rows and columns are periodic (donut structure)
    int periods[2] = {1, 1};
	
	//Definition of a new comunicator (for the elements in the grid)
    MPI_Comm grid_comm;
	
	//Definition of the new grid. 'MPI_COMM_WORLD' start communicator, '2' grid dimension, 'periods' periodicity
	//'1' flag to allow MPI to sort rank, '&grid_comm' pointer of the cartesian grid
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
	
	//Initialization of an array for row and column coordinates
	int coords[2];
	
	//Function to obtain the coordinates of a process. 'grid_comm' communicator of the cartesian grid, 'rank' of the process
	//of which we want the coordinates, '2' dimensions of the grid and 'coords' where the coordinates will be written
    MPI_Cart_coords(grid_comm, rank, 2, coords);
	
	//Initialization of the variables for the row coordinate and the columns coordinate
    int my_row = coords[0];
    int my_col = coords[1];
	
	//Initialization of the variables in which we save the rank of the neighboring processes
	int left, right, up, down;
	
	//Functions to find the rank of the neighboring processes in the cartesian grid. First '1' or '0' is the row/column
	//dimension, '-1' the shift direction. Other are pointers, the function inverts the direction of shifting for the second value
    MPI_Cart_shift(grid_comm, 1, -1, &right, &left); 
    MPI_Cart_shift(grid_comm, 0, -1, &down, &up);    
	
	
	//Initialization of a temporal block, used as buffer in the shifting of the blocks among the processes 
	double *tmp_block = malloc(block_size * block_size * sizeof(double));
	
	//This cycle is used for the initial alignment of the blocks in the A matrix. Every block must receive
	//the block of A that is designed for its multiplication. This is obtained shifting the blocks to the left, many times as 	
	//the number of 'my_row' variable. For every iteration the block is sent to the left and is received from the right. We use 
	//(my_col+my_row)%2 to avoid deadlock (every process wait and no one send). Once the PE receives its block it copies it in the A_block
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
	//Function to overwrite A_block with what is save in tmp_block 
    memcpy(A_block, tmp_block, block_size*block_size*sizeof(double));
	}
	
	//Same as above, but with matrix B 
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
	//Free dynamic memory of tmp_block
	free(tmp_block);
	
	
	//Cycle with k because so every block do the operation below and then exhange block with neighbors. 
	//Iteration on i and j calculates the moltiplication between local A_block and B_block and sum the result
	//in the local C_block
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
	
		//Initialization of two arrays that are used as buffer to receive the blocks of A and B during
		//the exchange process
        double *tmp_A = malloc(block_size * block_size * sizeof(double));
		double *tmp_B = malloc(block_size * block_size * sizeof(double));

		//Initialization of a status variable, for the receive process
		MPI_Status statusA;
		
		//Exchange of the blocks of A between the neigboring PEs in the grid. It shifts the process to the left. 
		//Try to avoid deadlock with the parity control (my_col+my_row)%2==0, such that there is no deadlock.
		//statusA can have different information, such as the source, the rank of the source process and 
		//a code for the error
		if ((my_col+my_row) % 2 == 0) {
			MPI_Send(A_block, block_size*block_size, MPI_DOUBLE, left, 0, grid_comm);
			MPI_Recv(tmp_A, block_size*block_size, MPI_DOUBLE, right, 0, grid_comm, &statusA);
			}    
		else {
			MPI_Recv(tmp_A, block_size*block_size, MPI_DOUBLE, right, 0, grid_comm, &statusA);
			MPI_Send(A_block, block_size*block_size, MPI_DOUBLE, left, 0, grid_comm);
			}
		memcpy(A_block, tmp_A, block_size*block_size*sizeof(double));
		
		
		// Dual of A but with B blocks
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
	
	//This part of code was only use in the debugging, to know how every process was working   
//	for (int r = 0; r < size; r++) {
//    if (rank == r) {
//        printf("Rank %d, C_block:\n", rank);
//        for (int i = 0; i < block_size; i++) {
//            for (int j = 0; j < block_size; j++) {
//                printf("%8.1f ", C_block[i*block_size+j]);
//				}
//            printf("\n");
//			}
//        fflush(stdout); //To print immediately
//		}
//    MPI_Barrier(MPI_COMM_WORLD);
//	}
	
	//FUnction to gather all the results for matrix C from all the MPI processes. 'C_block' pointer at the local block,
	//size, type of data, 'c' pointer at the global matrix, number of blocks to send, displs offset (position where the global
	//matrix has to insert the blocks), 'blocktype2' type of data, '0' rank of the process that gathers the data and communicator.
	MPI_Gatherv(C_block, block_size * block_size, MPI_DOUBLE, C, counts, displs, blocktype2, 0, MPI_COMM_WORLD);
	
	
	//Call of the function to write the matrix in an output file .csv
	if (rank == 0) {
		write_matrix_to_csv(argv[3], C, N);
	}
	
	//Memory and MPI freeing
    free(A_block); free(B_block); free(C_block);
    free(counts); free(displs);
    MPI_Type_free(&blocktype2);
    MPI_Comm_free(&grid_comm);
	
	//Take the value of the end time with process zero and calculate the elapsed time.
	//Under is to write the time in an output file for time
	if (rank == 0) {
		end_time = MPI_Wtime();
		elapsed_time = end_time - start_time;
		FILE *fp = fopen(argv[4], "w"); 
		if (fp) {
			fprintf(fp, "%d,%.6f\n", size, elapsed_time);
			fclose(fp);
		} else {
			perror("Error openening time file");
		}
	}
	
	//MPI finalization
    MPI_Finalize();	
    return 0;	
}

//Function to read the input matrices
void read_matrix_from_csv(const char *filename, double *matrix, int n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening input matrix file");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int result = fscanf(fp, "%lf,", &matrix[i*n + j]);
            if (result != 1) {
                fprintf(stderr, "Error reading element [%d,%d]\n", i, j);
                fclose(fp);
                exit(1);
            }
        }
    }

    fclose(fp);
}

//Function to write the output matrix
void write_matrix_to_csv(const char *filename, double *matrix, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error opening output matrix file");
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