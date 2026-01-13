#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Dimensione della matrice quadrata */
#define N 500 

/* Prototipi delle funzioni per leggere/scrivere le matrici da/verso CSV */
void read_matrix_from_csv();
void write_matrix_to_csv();



int main(int argc, char *argv[]){
	
	/* Rank, numero di processi e variabili per misurare il tempo */
	int rank, size;
	double start_time, end_time, elapsed_time;
	
	
	/* Inizializzazione MPI */ 
	MPI_Init(&argc, &argv);
	
	/* Recupero del rank del processo in MPI_COMM_WORLD */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	/* Recupero del numero totale di processi in MPI_COMM_WORLD */
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	/* Controllo argomenti in input (eseguibile, file A, file B, file output, file tempi).
	   Se mancano, chiudo. */
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
	
	/* p = radice quadrata del numero di processi (griglia p x p) */
	int p = (int)sqrt(size);	
	
	/* Verifico che il numero di processi sia un quadrato perfetto */
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
	
	/* Allocazione dinamica dei blocchi locali */
	double *A_block = malloc(block_size * block_size * sizeof(double));	
	double *B_block = malloc(block_size * block_size * sizeof(double));
	double *C_block = calloc(block_size * block_size, sizeof(double));	
	
	/* Puntatori alle matrici globali (solo su rank 0) */
	double *A=NULL, *B=NULL, *C=NULL;
	
	/* Rank 0 alloca le matrici globali e legge A e B dai CSV */
	if (rank==0){
		A = malloc(N * N * sizeof(double));
		B = malloc(N * N * sizeof(double));
		C = malloc(N * N * sizeof(double));
		read_matrix_from_csv(argv[1],A,N);
		read_matrix_from_csv(argv[2],B,N);
	}
	
	
	/* counts: numero di blocchi per processo (qui sempre 1).
	   displs: offset nella matrice globale da cui prelevare/inserire il blocco. */
	int *counts = malloc(size * sizeof(int));	
	int *displs = malloc(size * sizeof(int));	
	
	/* Riempio counts e displs scorrendo la griglia p x p */
	for (int i=0,idx=0; i<p; i++){
		for (int j=0; j<p; j++, idx++){
			counts[idx] = 1;
			displs[idx] = i * N * block_size + j * block_size;	
		}
	}
	
	/* Definizione di un tipo MPI per rappresentare un blocco della matrice */
	MPI_Datatype blocktype1, blocktype2;
	
	/* Creo un tipo "vettore" che descrive un blocco block_size x block_size dentro una matrice NxN */
	MPI_Type_vector(block_size, block_size, N, MPI_DOUBLE, &blocktype1);
		
	/* Ridimensiono il tipo per far sì che i blocchi siano contigui ai fini di Scatterv/Gatherv */
	MPI_Type_create_resized(blocktype1, 0, sizeof(double), &blocktype2);
	
	/* Registro il tipo nel sistema MPI */
	MPI_Type_commit(&blocktype2);	
	
	/* Distribuzione dei blocchi di A e B dal rank 0 a tutti i processi */
	MPI_Scatterv(A, counts, displs, blocktype2, A_block, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, counts, displs, blocktype2, B_block, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	
    
    
    
    
    
    
    /* CREAZIONE DELLA GRIGLIA CARTESIANA - Cannon (1)*/
	
	/* Dimensioni della griglia 2D: p righe e p colonne */
	int dims[2] = {p, p};
	
	/* Periodicità su entrambe le dimensioni (topologia toroidale) */
    int periods[2] = {1, 1};
	
	/* Nuovo comunicatore associato alla griglia */
    MPI_Comm grid_comm;
	
	/* Costruisco la griglia cartesiana */
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
	
	/* Coordinate (riga, colonna) del processo nella griglia */
	int coords[2];
	
	/* Recupero le coordinate del processo corrente */
    MPI_Cart_coords(grid_comm, rank, 2, coords);
	
	/* Indici di riga e colonna del processo */
    int my_row = coords[0];
    int my_col = coords[1];
	
	/* Rank dei vicini nella griglia (sinistra/destra/su/giu) */
	int left, right, up, down;
	
	/* Determino i vicini con uno shift nelle due dimensioni */
    MPI_Cart_shift(grid_comm, 1, -1, &right, &left); 
    MPI_Cart_shift(grid_comm, 0, -1, &down, &up);    
	
	/* Buffer temporaneo per gli shift iniziali */
	double *tmp_block = malloc(block_size * block_size * sizeof(double));
	
	
    
    
    
    /* SKEWING - Cannon (2)*/

    
    /* Allineamento iniziale dei blocchi di A: shift a sinistra di my_row passi */
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
	/* Aggiorno A_block con il blocco appena ricevuto */
    memcpy(A_block, tmp_block, block_size*block_size*sizeof(double));
	}

    /* Allineamento iniziale dei blocchi di B: shift verso l'alto di my_col passi */
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
	
    
    /* Libero il buffer temporaneo */
	free(tmp_block);
	



    


    /* MULTIPLICATION & SHIFTING - Cannon (3)*/

	/* Ciclo principale: calcolo locale e scambio blocchi tra vicini */
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
	
		/* Buffer di appoggio per ricevere i nuovi blocchi durante lo scambio */
        double *tmp_A = malloc(block_size * block_size * sizeof(double));
		double *tmp_B = malloc(block_size * block_size * sizeof(double));

		/* Stato per la ricezione */
		MPI_Status statusA;
		
		/* Scambio dei blocchi di A verso sinistra (parità per evitare deadlock) */
		if ((my_col+my_row) % 2 == 0) {
			MPI_Send(A_block, block_size*block_size, MPI_DOUBLE, left, 0, grid_comm);
			MPI_Recv(tmp_A, block_size*block_size, MPI_DOUBLE, right, 0, grid_comm, &statusA);
			}    
		else {
			MPI_Recv(tmp_A, block_size*block_size, MPI_DOUBLE, right, 0, grid_comm, &statusA);
			MPI_Send(A_block, block_size*block_size, MPI_DOUBLE, left, 0, grid_comm);
			}
		memcpy(A_block, tmp_A, block_size*block_size*sizeof(double));
		
		
		/* Scambio dei blocchi di B verso l'alto */
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
	
	/* Debug: stampa dei blocchi locali (lasciato commentato) */
//	for (int r = 0; r < size; r++) {
//    if (rank == r) {
//        printf("Rank %d, C_block:\n", rank);
//        for (int i = 0; i < block_size; i++) {
//            for (int j = 0; j < block_size; j++) {
//                printf("%8.1f ", C_block[i*block_size+j]);
//				}
//            printf("\n");
//			}
//        fflush(stdout);
//		}
//    MPI_Barrier(MPI_COMM_WORLD);
//	}
	
	/* Raccolgo i blocchi di C nel rank 0 per ricostruire la matrice globale */
	MPI_Gatherv(C_block, block_size * block_size, MPI_DOUBLE, C, counts, displs, blocktype2, 0, MPI_COMM_WORLD);
	
	
	/* Rank 0 scrive la matrice risultato su file CSV */
	if (rank == 0) {
		write_matrix_to_csv(argv[3], C, N);
	}
	
	/* Pulizia memoria e risorse MPI */
    free(A_block); free(B_block); free(C_block);
    free(counts); free(displs);
    MPI_Type_free(&blocktype2);
    MPI_Comm_free(&grid_comm);
	
	/* Rank 0 calcola il tempo totale e lo salva su file */
	if (rank == 0) {
		end_time = MPI_Wtime();
		elapsed_time = end_time - start_time;
		FILE *fp = fopen(argv[4], "w"); 
		if (fp) {
			fprintf(fp, "%d,%.6f\n", size, elapsed_time);
			fclose(fp);
		} else {
			perror("Errore apertura file tempi");
		}
	}
	
	/* Chiusura MPI */
    MPI_Finalize();	
    return 0;	
}





/* Funzioni di Supporto */




/* Lettura matrici di input */
void read_matrix_from_csv(const char *filename, double *matrix, int n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Errore apertura file matrice di input");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int result = fscanf(fp, "%lf,", &matrix[i*n + j]);
            if (result != 1) {
                fprintf(stderr, "Errore lettura elemento [%d,%d]\n", i, j);
                fclose(fp);
                exit(1);
            }
        }
    }

    fclose(fp);
}

/* Scrittura matrice di output */
void write_matrix_to_csv(const char *filename, double *matrix, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Errore apertura file matrice di output");
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
