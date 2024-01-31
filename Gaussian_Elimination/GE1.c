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
    double A_inv;
    int i,j,k;
    for(i=0;i<n;i++){
        A_inv = 1/A[i*n+i];
        for(j=i+1;j<n;j++){coeff[j] = A[i*n+j]*A_inv;}
        for(j=i+1;j<n;j++){A[i*n+j]=0;}
        for(j=i+1;j<n;j++){
            for(k=i+1;k<n;k++){
                A[j*n+k] -= coeff[k]*A[j*n+i];
            }
        }
    }
    return A;
}

int main(){
    unsigned long t2;
    unsigned long t1;
    int n=1000; int k = 6;
    int N=n;
    if(n>k){N=k;}
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