#include<stdio.h>
#include<omp.h>
#include<math.h>
#include<stdlib.h>
#include <sys/time.h>

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
double inline PROJ(double* arr1, double* arr2, int n){
    double proj = 0;
    for(int i=0;i<n;i++){proj+=arr1[i]*arr2[i];}
    return proj;
}

double* MATRIX_INV(double* array, int n){
    //GRAM SCHMIDT ORTHOGONALIZATION
    double* Q = (double*)calloc(n*n,sizeof(double));
    double* R = (double*)calloc(n*n,sizeof(double));
    double* Rinv = (double*)calloc(n*n,sizeof(double));
    double* Ainv = (double*)calloc(n*n,sizeof(double));
    double norm,proj;
    double coeff;
    int i,j,k;

    unsigned long t2;
    unsigned long t1;

    norm = 0;
    for(j=0;j<n;j++){norm += array[j]*array[j];}
    norm = sqrt(norm);
    for(j=0;j<n;j++){Q[j]=array[j]/norm;}

    for(i=1;i<n;i++){
        for(j=0;j<i;j++){
            proj = PROJ(&array[i*n],&Q[j*n],n);
            for(k=0;k<n;k++){Q[i*n+k] -= proj*Q[j*n+k];}
        }
        norm = 0;
        for(j=0;j<n;j++){
            Q[i*n+j]+=array[i*n+j];
            norm += Q[i*n+j]*Q[i*n+j];
        }
        norm = sqrt(norm);
        for(j=0;j<n;j++){Q[i*n+j]= Q[i*n+j]/norm;}
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

    return 0;
}