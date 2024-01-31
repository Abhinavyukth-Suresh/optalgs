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
    double Buff1;
    int i,j,k,l;
    for(i=0;i<n;i++){
        l = n-i-1;
        Buff1 = 1/A[i*n+i];
        SIMDptr1 = &A[i*n];
        #pragma omp simd
        for(j=i+1;j<n;j++){coeff[j] = SIMDptr1[j]*Buff1;}
        #pragma omp simd
        for(j=i+1;j<n;j++){SIMDptr1[j]=0;}
        for(j=i+1;j<n;j++){
            SIMDptr1 = &A[j*n+i+1];
            SIMDptr2 = &coeff[i+1];
            Buff1 = A[j*n+i];
            #pragma omp target enter data map(to:SIMDptr1[0:l],SIMDptr2[0:l],Buff1)
            #pragma omp target
            for(k=0;k<l;k++){
                SIMDptr1[k] -= SIMDptr2[k]*Buff1;
            }
            #pragma omp target exit data map(from:SIMDptr1[0:l])
        }
    }
    return A;
}

int main(){
    unsigned long t2;
    unsigned long t1;
    int n=3; int K = 6;
    int N=n;
    if(n>K){N=K;}
    double* A = (double*)calloc(n*n,sizeof(double));
    for(int i=0;i<n*n;i++){A[i]=(double)rand()/RAND_MAX;}
    double k[9] = {1,5,7,2,9,5,3,1,8};for(int i=0;i<n*n;i++){A[i]=k[i];}
    print_matrix(A,N,n);
    t1 = gettime();
    A=GE(A,n);
    t2 = gettime();
    printf("time taken %f s\n",((double)t2-t1)/1000000);
    printf("\n");
    print_matrix(A,N,n);
    return 0;
}