#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MSIZE 1024
#define T_A 250.0
#define T_B 540.0
#define T_env  15.0
#define Wx 0.3
#define Wy 0.2
#define PT 10
int ITER = 10000;


void init_matrix_a(double *M);
void init_matrix_b(double *M);
void init_zero(double *M);
void print_matrix(double *M);
void fprint_matrix(double *M, FILE *fp);
void run_diffusion_a(double *M, double *N);
void run_diffusion_b(double *M, double *N);
void isotropic_nv(double *M, double *N, int j, int k);
void anisotropic_nv(double *M, double *N, int j, int k);
void print_usage (void);



int main(int argc, char *argv[]) {

    if(argc > 3 || argc <2){

        print_usage();
        return(1);
    }

    if(argc == 3)
        ITER = atoi(argv[2]);

    double *M = malloc(MSIZE*MSIZE*sizeof(long double));
    if(M == NULL){
        perror("malloc");
        return 1;
    }

    double *N = malloc(MSIZE*MSIZE*sizeof(long double));
    if(N == NULL){
        perror("malloc");
        return 1;
    }
    init_zero(N);


    switch (argv[1][0]){

        case 'a' :

            init_matrix_a(M);
            // print_matrix(M);
            run_diffusion_a(M, N);   
            break;

        case 'b' :

            init_matrix_b(M);
            // print_matrix(M);
            run_diffusion_b(M, N);   
            break;

        default :

            print_usage();            
    }

    free(M);
    free(N);

    return 0;
}

void init_matrix_a (double *M){

    #pragma omp parrallel for collapse(2)
    for(int i=0; i < MSIZE; i++)
        for (int j = 0; j < MSIZE; j++) 
            if (j < MSIZE/2)
                M[i*MSIZE+j] = T_A;
            else 
                M[i*MSIZE+j] = T_env;
    return;
}

void init_matrix_b (double *M){

    #pragma omp parrallel for collapse(2)
    for(int i=0; i < MSIZE; i++)
        for (int j = 0; j < MSIZE; j++) 
            if (j >= MSIZE/4 && j < MSIZE*3/4 && i >= MSIZE/4 && i < MSIZE*3/4)
                M[i*MSIZE+j] = T_B;
            else 
                M[i*MSIZE+j] = T_env;
    return;
}

void init_zero(double *M){

    #pragma omp parallel for collapse(2)
    for(int i=0; i < MSIZE; i++)
        for (int j = 0; j < MSIZE; j++) 
            M[i*MSIZE+j] = 0.0;
    return;
}

void print_matrix (double *M){

    for (int j = 0; j < MSIZE; j++) {
        for (int k = 0; k < MSIZE; k++) {
            printf("%f ", M[j*MSIZE+k]);
        }
        printf("\n");
    }
    printf("\n");
    return;
}

void fprint_matrix(double *M, FILE *fp) {

    for (int j = 0; j < MSIZE; j++) {
        for (int k = 0; k < MSIZE; k++) {
            fprintf(fp, "%f ", M[j*MSIZE+k]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
    return;
}


void run_diffusion_a(double *M, double *N){

    FILE *time_fp, *temp_fp;
    char time_fn[32];
    int i, j, k, num_th, cond;
    double itime, ftime, exec_time = 0.0;

    num_th = atoi(getenv("OMP_NUM_THREADS"));
    cond   = num_th == PT && ITER > 99999;

    //open files to write data
    sprintf(time_fn, "./data/time_conf_a_%d", ITER);
    time_fp = fopen(time_fn, "a");
    if (time_fp == NULL){
        printf("Cannot open time_fp\n");
        return;
    }

    if (cond){

        temp_fp = fopen("./data/temp_conf_a", "w");
        if (time_fp == NULL){
            printf("Cannot open time_fp\n");
            return;
        }
        fprint_matrix(M, temp_fp);
    }
    
    for (i=0; i < ITER; i++){
        
        itime = omp_get_wtime();
        #pragma omp parallel for collapse(2)
        for (j = 0; j < MSIZE; j++) {
            for (k = 0; k < MSIZE; k++) {

                if(i%2 == 0)        
                    isotropic_nv(M, N, j, k);
                else
                    isotropic_nv(N, M, j, k);
            }
        }

        ftime = omp_get_wtime();
        exec_time += ftime - itime;

        if ((i+1)%2500 == 0 && cond)
            fprint_matrix(N, temp_fp);
    }

    fprintf(time_fp, "%d %f\n", num_th, exec_time);
    fclose(time_fp);

    if (i%100 != 0 && cond) {

        if (i%2 == 0)
            fprint_matrix(N, temp_fp);
        else 
            fprint_matrix(M, temp_fp);
        fclose(temp_fp);
    }
    return;
}

void isotropic_nv (double *M, double *N, int j, int k) {

    if (j == 0)

        if (k == 0)
            N[j*MSIZE+k] = (M[j*MSIZE+(k+1)] + M[j*MSIZE+k] + M[(j+1)*MSIZE+k] + M[j*MSIZE+k]) / 4;
        else if (k == (MSIZE - 1))
            N[j*MSIZE+k] = (M[j*MSIZE+(k-1)] + M[j*MSIZE+k] + M[(j+1)*MSIZE+k] + M[j*MSIZE+k]) / 4;
        else
            N[j*MSIZE+k] = (M[j*MSIZE+(k-1)] + M[j*MSIZE+(k+1)] + M[(j+1)*MSIZE+k] + M[j*MSIZE+k]) / 4;

    else if (k == 0)

        if (j == (MSIZE - 1))
            N[j*MSIZE+k] = (M[j*MSIZE+(k+1)] + M[j*MSIZE+k] + M[(j-1)*MSIZE+k] + M[j*MSIZE+k]) / 4;
        else
            N[j*MSIZE+k] = (M[j*MSIZE+(k+1)] + M[j*MSIZE+k] + M[(j+1)*MSIZE+k] + M[(j-1)*MSIZE+k]) / 4;

    else if (j == (MSIZE - 1))

        if (k == (MSIZE - 1))
            N[j*MSIZE+k] = (M[j*MSIZE+(k-1)] + M[j*MSIZE+k] + M[(j-1)*MSIZE+k] + M[j*MSIZE+k]) / 4;
        else
            N[j*MSIZE+k] = (M[j*MSIZE+(k-1)] + M[j*MSIZE+(k+1)] + M[(j-1)*MSIZE+k] + M[j*MSIZE+k]) / 4;

    else if (k == (MSIZE - 1))

        N[j*MSIZE+k] = (M[j*MSIZE+(k-1)] + M[j*MSIZE+k] + M[(j+1)*MSIZE+k] + M[(j-1)*MSIZE+k]) / 4;

    else    

        N[j*MSIZE+k] = (M[j*MSIZE+(k+1)] + M[j*MSIZE+(k-1)] + M[(j+1)*MSIZE+k] + M[(j-1)*MSIZE+k]) / 4;

    return;    
}

void run_diffusion_b(double *M, double *N){

    FILE *time_fp, *temp_fp;
    char time_fn[32];
    int i, j, k, num_th, cond;
    double itime, ftime, exec_time = 0.0;

    num_th = atoi(getenv("OMP_NUM_THREADS"));
    cond   = num_th == PT && ITER > 999999;

    //open files to write data
    sprintf(time_fn, "./data/time_conf_b_%d", ITER);
    time_fp = fopen(time_fn, "a");
    if (time_fp == NULL){
        printf("Cannot open time_fp\n");
        return;
    }

    if (cond){

        temp_fp = fopen("./data/temp_conf_b", "w");
        if (time_fp == NULL){
            printf("Cannot open time_fp\n");
            return;
        }
        fprint_matrix(M, temp_fp);
    }
    
    for (i=0; i < ITER; i++){
        
        itime = omp_get_wtime();
        #pragma omp parallel for collapse(2)
        for (j = 0; j < MSIZE; j++) {
            for (k = 0; k < MSIZE; k++) {

                if(i%2 == 0)        
                    anisotropic_nv(M, N, j, k);
                else
                    anisotropic_nv(N, M, j, k);
            }
        }

        ftime = omp_get_wtime();
        exec_time += ftime - itime;

        if ((i+1)%2500 == 0 && cond)
            fprint_matrix(N, temp_fp);
    }

    fprintf(time_fp, "%d %f\n", num_th, exec_time);
    fclose(time_fp);

    if (i%100 != 0 && cond) {
        
        if (i%2 == 0)
            fprint_matrix(N, temp_fp);
        else 
            fprint_matrix(M, temp_fp);
        fclose(temp_fp);
    }
    return;
}

void anisotropic_nv(double *M, double *N, int j, int k) {

    if (j == 0)

        if (k == 0)
            N[j*MSIZE+k] = Wx * (M[j*MSIZE+(k+1)] + M[j*MSIZE+k]) + Wy * (M[(j+1)*MSIZE+k] + M[j*MSIZE+k]);
        else if (k == (MSIZE - 1))
            N[j*MSIZE+k] = Wx * (M[j*MSIZE+(k-1)] + M[j*MSIZE+k]) + Wy * (M[(j+1)*MSIZE+k] + M[j*MSIZE+k]);
        else
            N[j*MSIZE+k] = Wx * (M[j*MSIZE+(k-1)] + M[j*MSIZE+(k+1)]) + Wy * (M[(j+1)*MSIZE+k] + M[j*MSIZE+k]);

    else if (k == 0)

        if (j == (MSIZE - 1))
            N[j*MSIZE+k] = Wx * (M[j*MSIZE+(k+1)] + M[j*MSIZE+k]) + Wy * (M[(j-1)*MSIZE+k] + M[j*MSIZE+k]);
        else
            N[j*MSIZE+k] = Wx *  (M[j*MSIZE+(k+1)] + M[j*MSIZE+k]) + Wy * (M[(j+1)*MSIZE+k] + M[(j-1)*MSIZE+k]);

    else if (j == (MSIZE - 1))

        if (k == (MSIZE - 1))
            N[j*MSIZE+k] = Wx * (M[j*MSIZE+(k-1)] + M[j*MSIZE+k]) + Wy * (M[(j-1)*MSIZE+k] + M[j*MSIZE+k]);
        else
            N[j*MSIZE+k] = Wx * (M[j*MSIZE+(k-1)] + M[j*MSIZE+(k+1)]) + Wy * (M[(j-1)*MSIZE+k] + M[j*MSIZE+k]);

    else if (k == (MSIZE - 1))

        N[j*MSIZE+k] = Wx * (M[j*MSIZE+(k-1)] + M[j*MSIZE+k]) + Wy * (M[(j+1)*MSIZE+k] + M[(j-1)*MSIZE+k]);

    else    

        N[j*MSIZE+k] = Wx * (M[j*MSIZE+(k+1)] + M[j*MSIZE+(k-1)]) + Wy * (M[(j+1)*MSIZE+k] + M[(j-1)*MSIZE+k]);

    return;
}

void print_usage (void){

    fprintf(stderr, "Invalid argument!\n");
    fprintf(stdout, "Usage: OMP_NUM_THREADS=[numthreds] ./Heat_diff [a|b] [num iteration]\n");
    return;
}
