#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Dimensione (righe/colonne) della matrice */
#define N 2000

/* Prototipi per lettura/scrittura CSV */
void read_matrix_from_csv();
void write_matrix_to_csv();



int main(int argc, char *argv[]){
	
	/* Rank, numero processi e variabili per il tempo */
	int rank, size;
	double start_time, end_time, elapsed_time;
	
	
	/* Avvio MPI */ 
	MPI_Init(&argc, &argv);
	
	/* Recupero rank e numero totale di processi */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	/* p = radice del numero di processi (griglia p x p) */
	int p = (int)sqrt(size);
	
	/* Controllo argomenti: eseguibile + file A + file B + file output + file tempi */
	if (argc < 5) {
		if (rank == 0) {
			printf("Dati di input mancanti: %s\n", argv[0]);
		}
		MPI_Finalize();
		return 1;
	}
		
	
	/* Tempo iniziale misurato dal processo 0 */
	if (rank == 0) {
		start_time = MPI_Wtime();
	}
	
	


	/* Verifico che size sia un quadrato perfetto */
	if (p*p!=size){
		if (rank==0){
			printf("Il numero di processi deve essere un quadrato perfetto.\n");
		}
		MPI_Finalize();
		return 1;
	}
	
	


	
	/* Verifico che N sia divisibile per p, così posso suddividere NxN in p*p blocchi */
	if (N%p!=0){
		if (rank==0){
			printf("La dimensione della matrice deve essere divisibile per p = %d\n", p);
		}
		MPI_Finalize();
		return 1;
	}

	/* Dimensione del blocco locale (righe/colonne per ciascun PE) */
	int block_size = N/p; 
	
	/* Allocazione dei blocchi locali */
	double *A_block = malloc(block_size * block_size * sizeof(double));	
	double *B_block = malloc(block_size * block_size * sizeof(double));
	double *C_block = calloc(block_size * block_size, sizeof(double));	
	
	/* Matrici globali (solo su rank 0) */
	double *A=NULL, *B=NULL, *C=NULL;
	
	
	/* Rank 0 alloca e legge A e B */
	if (rank==0){
		A = malloc(N * N * sizeof(double));
		B = malloc(N * N * sizeof(double));
		C = malloc(N * N * sizeof(double));
		read_matrix_from_csv(argv[1],A,N);
		read_matrix_from_csv(argv[2],B,N);
	}
	
	
	/* counts: un blocco per processo; displs: offset dei blocchi nella matrice globale */
	int *counts = malloc(size * sizeof(int));	
	int *displs = malloc(size * sizeof(int));	
	
	/* Inizializzo counts e displs scorrendo la griglia p x p */
	for (int i=0,idx=0; i<p; i++){
		for (int j=0; j<p; j++, idx++){
			counts[idx] = 1;
			displs[idx] = i * N * block_size + j * block_size;	
		}
	}
	
	
	/* Tipo MPI per descrivere un blocco block_size x block_size dentro una matrice NxN */
	MPI_Datatype blocktype1, blocktype2;
	MPI_Type_vector(block_size, block_size, N, MPI_DOUBLE, &blocktype1);	
	MPI_Type_create_resized(blocktype1, 0, sizeof(double), &blocktype2);
	MPI_Type_commit(&blocktype2);	
	
	
	/* Distribuzione dei blocchi di A e B ai processi */
	MPI_Scatterv(A, counts, displs, blocktype2, A_block, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    	MPI_Scatterv(B, counts, displs, blocktype2, B_block, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	
	
	/* Creazione griglia cartesiana 2D con periodicità */
	int dims[2] = {p, p};
    int periods[2] = {1, 1};
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
	
	/* Coordinate (riga, colonna) del processo */
	int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];
	
	/* Rank dei vicini nella griglia */
	int left, right, up, down;
    MPI_Cart_shift(grid_comm, 1, -1, &right, &left); 
    MPI_Cart_shift(grid_comm, 0, -1, &down, &up);    
	
	
	
	/* Buffer temporaneo per gli allineamenti iniziali */
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
	
	
	
	/* Ciclo principale: calcolo locale e scambio blocchi */
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
	
	
        /* Buffer di appoggio per ricevere i nuovi blocchi */
        double *tmp_A = malloc(block_size * block_size * sizeof(double));
	double *tmp_B = malloc(block_size * block_size * sizeof(double));

		
		/* Scambio del blocco A verso sinistra */
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
	
		/* Scambio del blocco B verso l'alto */
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

	
	/* Raccolta dei blocchi di C nel processo 0 */
	MPI_Gatherv(C_block, block_size * block_size, MPI_DOUBLE, C, counts, displs, blocktype2, 0, MPI_COMM_WORLD);

	/* Scrittura della matrice finale */
	if (rank == 0) {
		write_matrix_to_csv(argv[3], C, N);
	}
	
	
    /* Pulizia memoria e risorse MPI */
    free(A_block); free(B_block); free(C_block);
    free(counts); free(displs);
    MPI_Type_free(&blocktype2);
    MPI_Comm_free(&grid_comm);
	
	/* Calcolo e salvataggio del tempo totale */
	if (rank == 0) {
		end_time = MPI_Wtime();
		elapsed_time = end_time - start_time;

		FILE *fp = fopen(argv[4], "w"); 
		if (fp) {
			fprintf(fp, "%d,%.6f\n", size, elapsed_time);
			fclose(fp);
		} else {
			perror("Errore nell'apertura del file tempi");
		}
	}
	
    /* Chiusura MPI */
    MPI_Finalize();
    return 0;	
}


/* Lettura matrice da file CSV */
void read_matrix_from_csv(const char *filename, double *matrix, int n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Errore nell'apertura del file matrice di input");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int result = fscanf(fp, "%lf,", &matrix[i*n + j]);
            if (result != 1) {
                fprintf(stderr, "Errore nella lettura dell'elemento [%d,%d]\n", i, j);
                fclose(fp);
                exit(1);
            }
        }
    }

    fclose(fp);
}

/* Scrittura matrice su file CSV */
void write_matrix_to_csv(const char *filename, double *matrix, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Errore nell'apertura del file matrice di output");
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
