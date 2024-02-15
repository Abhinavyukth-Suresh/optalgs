/*

Author : Abhinav Yukth S
         IMS21007

            GAUSSIAN ELIMINATION WITH PARTIAL PIVOTING
An optimised implementation of Gaussian elimination with partial pivoting.
The implementation, the memory arragnement of the array is done in column major format, 
ie the Fortran style, this allows for effiecient caching of data. This reduces the number
 of times the data is being copied from RAM, thus reducing hardware and os level protocols.
Further performance improvement is achieved by Vectorization, the use of SIMD, to parallelize
 multiple data for a single instruction in singl CPU. Rather than implementing specifially via
 imintrinsic functions like direclty calling ymm registers, here it is achieved via openmp, 
 this enables scope for native system specific compiler optimizations. 

*/

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include <sys/time.h>

double* GE(double* A,int n){
    double* coeff = (double*)calloc(n,sizeof(double));
    double *SIMDptr1, *SIMDptr2;
    double d_buff1,d_buff2;
    int i,j,k,i_buff1;

    for(i=0;i<n-1;i++){
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
            for(j=0;j<n+1;j++){
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
        for(j=i+1;j<n+1;j++){
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