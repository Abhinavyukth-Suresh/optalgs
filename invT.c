#include<stdio.h>
#include<omp.h>
#include<math.h>
#include<stdlib.h>
#include <sys/time.h>

//gcc -o test -msse -mavx2 -mavx -fopenmp -ffast-math -funroll-all-loops -O3 -DN_THREAD=7 -DN=1500 invT.c

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
    double *SIMDptr1,*SIMDptr2,*SIMDptr3,*SIMDptr4;
    double norm,proj;
    double coeff;
    int i,j,k;
    unsigned long t2;
    unsigned long t1;
	
    norm = 0;
    #pragma omp simd reduction(+:norm)
    for(j=0;j<n;j++){norm += array[j]*array[j];}
    norm = sqrt(norm);
    #pragma omp target enter data map(to:Q[0:n],array[0:n],norm)
    #pragma omp target
    for(j=0;j<n;j++){Q[j]=array[j]/norm;}
    #pragma omp target exit data map(from:Q[0:n])

    for(i=1;i<n;i++){
        SIMDptr1 = &Q[i*n];
        for(j=0;j<i;j++){
            SIMDptr2 = &Q[j*n];
            proj=0;
            #pragma omp simd reduction(+:proj)
            for(int k=0;k<n;k++){proj+=array[i*n+k]*SIMDptr2[k];}
            for(k=0;k<n;k++){SIMDptr1[k] -= proj*SIMDptr2[k];}       
        }
        norm = 0;
        #pragma omp simd
        for(j=0;j<n;j++){
            Q[i*n+j]+=array[i*n+j];   
        }
        #pragma omp simd reduction(+:norm)
        for(j=0;j<n;j++){
            norm += Q[i*n+j]*Q[i*n+j];
        }
        norm = sqrt(norm);
        #pragma omp simd
        for(j=0;j<n;j++){Q[i*n+j]= Q[i*n+j]/norm;}
    }
    
    for(i=0;i<n;i++){
        for(j=0;j<=i;j++){
            #pragma omp simd aligned(R,array,Q:64)
            for(k=0;k<n;k++){
                R[i*n+j] += array[i*n+k]*Q[j*n+k];               
            }
        }
    }
   
    //#pragma omp parallel for num_threads(N_THREAD)
    for(int i=0;i<n;i++){
        Rinv[i*n+i] = 1;
        coeff = 1/R[i*n+i];
        #pragma omp simd aligned(Rinv:64)
        for(int k=0;k<n;k++){
            Rinv[i*n+k] *= coeff;
        }
        #pragma omp simd aligned(R:64)
        for(int k=0;k<n;k++){
            R[i*n+k] *= coeff;
        }
    }
    //#pragma omp parallel for shared(R,Rinv) num_threads(N_THREAD)
    for(i=1;i<n;i++){
        SIMDptr1 = &R[i*n];
        SIMDptr3 = &Rinv[i*n];
        for(j=0;j<i;j++){
            coeff = R[i*n+j]/R[j*n+j];
            SIMDptr2 = &R[j*n];
            SIMDptr4 = &Rinv[j*n];
            //#pragma omp simd //aligned(SIMDptr1,SIMDptr2:64)
            #pragma omp target enter data map(to:SIMDptr1[0:n],SIMDptr2[0:n],coeff)
            #pragma omp target
            for(k=0;k<n;k++){
                SIMDptr1[k]-=coeff*SIMDptr2[k];
            }
            #pragma omp target exit data map(from:SIMDptr1[0:n])

            #pragma omp target enter data map(to:SIMDptr3[0:n],SIMDptr4[0:n],coeff)
            #pragma omp target
            for(k=0;k<n;k++){
                SIMDptr3[k]-=coeff*SIMDptr4[k];
            }
            #pragma omp target exit data map(from:SIMDptr3[0:n])
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
        matrix[k]=rand()*10/RAND_MAX;
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
