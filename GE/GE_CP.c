/*

Author : Abhinav Yukth S
         IMS21007

            GAUSSIAN ELIMINATION WITH COMPLETE PIVOTING
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
#include<math.h>

double* GE(double* A,int n){
    double* coeff = (double*)calloc(n,sizeof(double));
    double A_inv;
    double d_buff1,d_buff2,d_buff3;
    int i,j,k,i_buff1,i_buff2;

    d_buff2 = fabs(A[0]);
    i_buff1 = 0;
    i_buff2 = 0;
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            d_buff3 = fabs(A[i*n+j]);
            if(d_buff3>d_buff2){
                d_buff2 = d_buff3;
                i_buff1 = i;
                i_buff2 = j;
            }
        }
    }

    for(i=0;i<n;i++){
        if(i_buff1!=i){
            for(j=0;j<n;j++){
                d_buff1 = A[i*n+j];
                A[i*n+j] = A[i_buff1*n+j];
                A[i_buff1*n+j] = d_buff1;
            }
        }
        if(i_buff2!=j){
            for(j=0;j<n+1;j++){
                d_buff1 = A[j*n+i];
                A[j*n+i] = A[j*n+i_buff2];
                A[j*n+i_buff2] = d_buff1;
            }
        }
        A_inv = 1/A[i*n+i];
        #pragma omp simd
        for(j=i+1;j<n;j++){coeff[j] = A[i*n+j]*A_inv;}
        #pragma omp simd
        for(j=i+1;j<n;j++){A[i*n+j]=0;}
        d_buff2 = fabs(A[(i+1)*n+i+1]);
        i_buff1 = i+1;
        i_buff2 = i+1;
        for(j=i+1;j<n+1;j++){
            for(k=i+1;k<n;k++){          
                A[j*n+k] -= coeff[k]*A[j*n+i];
                d_buff3 = fabs(A[j*n+k]);
                if(d_buff3>d_buff2 && (j<n)){
                    d_buff2 = d_buff3;
                    i_buff1 = j;
                    i_buff2 = k;
                }
            }
        }
        
    }
    return A;
}