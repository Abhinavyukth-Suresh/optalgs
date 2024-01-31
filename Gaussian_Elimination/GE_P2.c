#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include <sys/time.h>

void print_matrix(double *A,int N,int n){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){printf(" %f ,",A[i*n+j]);}
        printf("\n");
    }
}

long gettime(){
	struct timeval currentTime;
	gettimeofday(&currentTime, NULL);
	return currentTime.tv_sec * (int)1e6 + currentTime.tv_usec;
}

double* GE(double* A,int n){
    double* coeff = (double*)calloc(n,sizeof(double));
    double *SIMDptr1, *SIMDptr2;
    double d_buff1,d_buff2;
    int i,j,k,i_buff1;

    for(i=0;i<n;i++){
        d_buff2 = A[i*n+i]>0?A[i*n+i]:-A[i*n+i];
        i_buff1 = i;
        for(j=i+1;j<n;j++){
            d_buff1 = (A[i*n+j]>0?A[i*n+j]:-A[i*n+j]);
            if(d_buff2<d_buff1){
                d_buff2 = d_buff1;
                i_buff1 = j;
            }
        }
        if(i!=i_buff1){
            for(j=0;j<n;j++){
                d_buff1 = A[j*n+i];
                A[j*n+i] = A[j*n+i_buff1];
                A[j*n+i_buff1] = d_buff1;
            }
        }

        i_buff1 = n-i-1;
        d_buff1 = 1/A[i*n+i];
        SIMDptr1 = &A[i*n];
        #pragma omp simd
        for(j=i+1;j<n;j++){coeff[j] = SIMDptr1[j]*d_buff1;}
        #pragma omp simd
        for(j=i+1;j<n;j++){SIMDptr1[j]=0;}
        for(j=i+1;j<n;j++){
            SIMDptr1 = &A[j*n+i+1];
            SIMDptr2 = &coeff[i+1];
            d_buff1 = A[j*n+i];
            #pragma omp target enter data map(to:SIMDptr1[0:i_buff1],SIMDptr2[0:i_buff1],d_buff1)
            #pragma omp target
            for(k=0;k<i_buff1;k++){
                SIMDptr1[k] -= SIMDptr2[k]*d_buff1;
            }
            #pragma omp target exit data map(from:SIMDptr1[0:i_buff1])
        }
    }
    return A;
}

int main(){
    unsigned long t2;
    unsigned long t1;
    int n=1000; int K = 6;
    int N=n;
    if(n>K){N=K;}
    double* A = (double*)calloc(n*n,sizeof(double));
    for(int i=0;i<n*n;i++){A[i]=(double)rand()/RAND_MAX;}
    //double k[9] = {1,5,7,2,9,5,3,1,8};for(int i=0;i<n*n;i++){A[i]=k[i];}
    print_matrix(A,N,n);
    t1 = gettime();
    A=GE(A,n);
    t2 = gettime();
    printf("time taken %f s\n",((double)t2-t1)/1000000);
    printf("\n");
    print_matrix(A,N,n);
    return 0;
}