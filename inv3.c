#include<stdio.h>
#include<omp.h>
#include<math.h>
#include<stdlib.h>
#include <sys/time.h>

//gcc -o test invT.c -ffast-math -O3 -msse -mavx2 -DN=1000 -fopenmp -DN_THREADS=7s

long gettime(){
	struct timeval currentTime;
	gettimeofday(&currentTime, NULL);
	return currentTime.tv_sec * (int)1e6 + currentTime.tv_usec;
}

#ifndef N
    #define N 50
#endif

#ifndef N_THREAD
    #define N_THREAD 4
#endif

double* MATRIX_INV(double* array, int n){
    
    double* Q = (double*)calloc(n*n,sizeof(double));
    double* R = (double*)calloc(n*n,sizeof(double));
    double* Rinv = (double*)calloc(n*n,sizeof(double));
    double* Ainv = (double*)calloc(n*n,sizeof(double));
    double *pointer1,*pointer2;
    double norm,proj;
    double simd[4];
    double coeff;
    int i,j,k;

    unsigned long t2;
    unsigned long t1;

    norm = 0;
    #pragma omp simd reduction(+:norm)
    for(j=0;j<n;j++){norm += array[j]*array[j];}
    norm = 1/sqrt(norm);
    #pragma omp target enter data map(to:Q[0:n],array[0:n],norm)
    #pragma omp target
    for(j=0;j<n;j++){Q[j]=array[j]*norm;}
    #pragma omp target exit data map(from:Q[0:n])

    for(i=1;i<n;i++){
        pointer1 = &Q[i*n];
        for(j=0;j<i;j++){
            pointer2 = &Q[j*n];
            proj=0;
            #pragma omp simd reduction(+:norm)
            for(k=0;k<n;k++){proj+=array[i*n+k]*Q[i*n+k];}
            #pragma omp target enter data map(to:pointer1[0:n],pointer2[0:n],proj)
            #pragma omp target
            for(k=0;k<n;k++){pointer1[k] -= proj*pointer2[k];}
            #pragma omp target exit data map(from:pointer1[0:n])
        }
        norm = 0;
        for(j=0;j<n;j++){
            Q[i*n+j]+=array[i*n+j];
            norm += Q[i*n+j]*Q[i*n+j];
        }
        norm = 1/sqrt(norm);
        for(j=0;j<n;j++){Q[i*n+j]= Q[i*n+j]*norm;}
    }

    for(i=0;i<n;i++){
        for(j=0;j<=i;j++){
            for(k=0;k<n;k++){
                R[i*n+j] += array[i*n+k]*Q[j*n+k];               
            }
        }
    }

    #pragma omp parallel for num_threads(N_THREAD)
    for(int i=0;i<n;i++){
        Rinv[i*n+i] = 1;
        coeff = 1/R[i*n+i];
        for(int k=0;k<n;k++){
            Rinv[i*n+k] *= coeff;
            R[i*n+k] *= coeff;
        }
    }
    #pragma omp parallel for num_threads(N_THREAD)
    for(i=1;i<n;i++){
        for(j=0;j<i;j++){
            coeff = R[i*n+j]/R[j*n+j];
            for(k=0;k<n;k++){
                R[i*n+k]-=coeff*R[j*n+k];
                Rinv[i*n+k]-=coeff*Rinv[j*n+k];
            }
        }
    }

    #pragma omp parallel for num_threads(N_THREAD)
    for(int k=0;k<n;k++){
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                Ainv[i*n+j]+=Q[k*n+i]*Rinv[k*n+j];
            }
        }
    }
    return Ainv;
}



int main(){
    int n = N;
    printf("n=%d\n",n);
    double* matrix = (double*)calloc(n*n,sizeof(double));

    for(int i=0;i<13*n;i++){
        double j = (double)rand()*(n*n)/RAND_MAX;
        int k = (int)j;
        matrix[k]=rand();
    }

    unsigned long t1 = gettime();
    double* inv = MATRIX_INV(matrix,n);
    unsigned long t2 = gettime();
    printf("time taken in us %d \n",t2-t1);
    printf("time taken %f s\n",((double)t2-t1)/1000000);
    for(int i=0;i<10;i++){
        printf("inv %f \n",inv[i]);
    }

    return 0;
}