#include <immintrin.h>
#include <stdlib.h>
#include <time.h>
#include<stdio.h>
#include<string.h>

#define k_b 512  // Panel width for blocking (fits in L1)
#define m_b 32   // Block size for A (fits in L2)
#define m_r 8    // Rows computed in registers
#define n_r 4    // Columns computed in registers (AVX2 handles 4 doubles per YMM)

void inner_kernel(double* __restrict__ hat_a,
                  double* __restrict__ hat_b,
                  double* __restrict__ hat_c,
                  int ldc) 
{
    __m256d R00 = _mm256_setzero_pd();
    __m256d R01 = _mm256_setzero_pd();
    __m256d R02 = _mm256_setzero_pd();
    __m256d R03 = _mm256_setzero_pd();

    for (int i = 0; i < k_b; ++i) 
    {
        // Prefetch next values to avoid stalls
        _mm_prefetch((char*)(hat_a + 16), _MM_HINT_T0);
        _mm_prefetch((char*)(hat_b + 16), _MM_HINT_T0);

        __m256d A0 = _mm256_set1_pd(*(hat_a + 0));
        __m256d A1 = _mm256_set1_pd(*(hat_a + 1));
        __m256d A2 = _mm256_set1_pd(*(hat_a + 2));
        __m256d A3 = _mm256_set1_pd(*(hat_a + 3));

        __m256d B = _mm256_load_pd(hat_b);

        asm volatile(
            "vfmadd231pd %[A0], %[B], %[R00]\n\t"
            "vfmadd231pd %[A1], %[B], %[R01]\n\t"
            "vfmadd231pd %[A2], %[B], %[R02]\n\t"
            "vfmadd231pd %[A3], %[B], %[R03]\n\t"
            : [R00] "+v" (R00),
              [R01] "+v" (R01),
              [R02] "+v" (R02),
              [R03] "+v" (R03)
            : [A0] "v" (A0),
              [A1] "v" (A1),
              [A2] "v" (A2),
              [A3] "v" (A3),
              [B] "v" (B)
            : "memory"
        );

        hat_a += m_r;
        hat_b += n_r;
    }

    // Store computed values back to C matrix
    _mm256_storeu_pd(hat_c + 0 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 0 * ldc), R00));
    _mm256_storeu_pd(hat_c + 1 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 1 * ldc), R01));
    _mm256_storeu_pd(hat_c + 2 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 2 * ldc), R02));
    _mm256_storeu_pd(hat_c + 3 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 3 * ldc), R03));
}

/**
 * @brief Optimized DGEMM for Tiger Lake (Blocked Implementation)
 */
void dgemm_tigerlake(int m, int k, int n,
                     double* src_a, double* src_b, double* src_c,
                     int lda, int ldb, int ldc) 
{
    double* pak_a = (double*)_mm_malloc(m_b * k_b * sizeof(double), 64);
    double* pak_b = (double*)_mm_malloc(k_b * n * sizeof(double), 64);

    for (int k_b_i = 0; k_b_i < k / k_b; k_b_i++) 
    {
        // Prefetch B into L1
        for (int i = 0; i < k_b; i++)
            _mm_prefetch((char*)(src_b + k_b_i * k_b * ldb + i * ldb), _MM_HINT_T0);

        for (int m_b_i = 0; m_b_i < m / m_b; m_b_i++) 
        {
            // Pack A into L2-friendly blocks
            for (int i = 0; i < m_b; i++)
                for (int j = 0; j < k_b; j++)
                    pak_a[i * k_b + j] = src_a[(m_b_i * m_b + i) * lda + (k_b_i * k_b + j)];

            for (int n_r_i = 0; n_r_i < n / n_r; n_r_i++) 
            {
                for (int m_r_i = 0; m_r_i < m_b / m_r; m_r_i++) 
                {
                    // Call optimized inner kernel
                    inner_kernel(pak_a + m_r_i * m_r * k_b,
                                 src_b + k_b_i * k_b * ldb + n_r_i * n_r * k_b,
                                 src_c + m_b_i * m_b * ldc + m_r_i * m_r * ldc + n_r_i * n_r,
                                 ldc);
                }
            }
        }
    }

    _mm_free(pak_a);
    _mm_free(pak_b);
}


void initialize_matrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = 1;//(double)(rand() % 100) / 10.0; // Random values between 0.0 and 10.0
    }
}

int main(){
    int M,N,K,l;
    l=1024;
    M = l;N=l;K=l;

    double* A = (double*)malloc(M * K * sizeof(double));
    double* B = (double*)malloc(K * N * sizeof(double));
    double* C = (double*)malloc(M * N * sizeof(double));


    initialize_matrix(A, M, K);
    initialize_matrix(B, K, N);
    memset(C, 0, M * N * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    dgemm_tigerlake(M,K,N,A,B,C,K,N,N);
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate the elapsed time in microseconds
    long elapsed_us = (end.tv_sec - start.tv_sec) * 1000000L +
                      (end.tv_nsec - start.tv_nsec) / 1000;
    printf("Function took %ld microseconds.\n", elapsed_us);
    printf(" %f, %f, %f \n",C[0],C[1],C[2]);
    return 0;
}
