#include <immintrin.h>
#include <stdlib.h>
#include <time.h>
#include<stdio.h>
#include<string.h>

#define TO_STRING_HELPER(X)   #X 
#define TO_STRING(X)          TO_STRING_HELPER(X)

#if defined(__ICC) || defined(__ICL)
  #define UNROLL_LOOP(n)      _Pragma(TO_STRING(unroll (n)))
#elif defined(__clang__)
  #define UNROLL_LOOP(n)      _Pragma(TO_STRING(unroll (n)))
#elif defined(__GNUC__) && !defined(__clang__)
  #define UNROLL_LOOP(n)      _Pragma(TO_STRING(GCC unroll (n)))
#elif defined(_MSC_BUILD)
  #pragma message ("Microsoft Visual C++ (MSVC) detected: Loop unrolling not supported!")
  #define UNROLL_LOOP(n)
#else
  #warning "Unknown compiler: Loop unrolling not supported!"
  #define UNROLL_LOOP(n)
#endif


#define k_b 512
#define m_b 32
#define m_r 12
#define n_r 4

inline void inner_kernel(double* __restrict__ hat_a, double* __restrict__ hat_b, double* __restrict__ hat_c, int ldc) {
    
    __m256d R00 = _mm256_setzero_pd();
    __m256d R01 = _mm256_setzero_pd();
    __m256d R02 = _mm256_setzero_pd();
    __m256d R03 = _mm256_setzero_pd();
    __m256d R04 = _mm256_setzero_pd();
    __m256d R05 = _mm256_setzero_pd();
    __m256d R06 = _mm256_setzero_pd();
    __m256d R07 = _mm256_setzero_pd();
    __m256d R08 = _mm256_setzero_pd();
    __m256d R09 = _mm256_setzero_pd();
    __m256d R10 = _mm256_setzero_pd();
    __m256d R11 = _mm256_setzero_pd();

    for (int i = 0; i < k_b; ++i) {
        _mm_prefetch((char*)(hat_a + 16), _MM_HINT_T0);
        _mm_prefetch((char*)(hat_b + 16), _MM_HINT_T0);

        __m256d B = _mm256_load_pd(hat_b);

        asm volatile(
            "vfmadd231pd 0(%[hat_a])%{1to4}, %[B], %[R00]\n\t"
            "vfmadd231pd 8(%[hat_a])%{1to4}, %[B], %[R01]\n\t"
            "vfmadd231pd 16(%[hat_a])%{1to4}, %[B], %[R02]\n\t"
            "vfmadd231pd 24(%[hat_a])%{1to4}, %[B], %[R03]\n\t"
            "vfmadd231pd 32(%[hat_a])%{1to4}, %[B], %[R04]\n\t"
            "vfmadd231pd 40(%[hat_a])%{1to4}, %[B], %[R05]\n\t"
            "vfmadd231pd 48(%[hat_a])%{1to4}, %[B], %[R06]\n\t"
            "vfmadd231pd 56(%[hat_a])%{1to4}, %[B], %[R07]\n\t"
            "vfmadd231pd 64(%[hat_a])%{1to4}, %[B], %[R08]\n\t"
            "vfmadd231pd 72(%[hat_a])%{1to4}, %[B], %[R09]\n\t"
            "vfmadd231pd 80(%[hat_a])%{1to4}, %[B], %[R10]\n\t"
            "vfmadd231pd 88(%[hat_a])%{1to4}, %[B], %[R11]\n\t"
            : [R00] "+v"(R00), [R01] "+v"(R01), [R02] "+v"(R02), [R03] "+v"(R03),
              [R04] "+v"(R04), [R05] "+v"(R05), [R06] "+v"(R06), [R07] "+v"(R07),
              [R08] "+v"(R08), [R09] "+v"(R09), [R10] "+v"(R10), [R11] "+v"(R11)
            : [B] "v"(B), [hat_a] "r"(hat_a)
            : "memory"
        );

        hat_a += m_r;
        hat_b += n_r;
    }

    //for (int i = 0; i < 12; i++) {
     //   _mm256_storeu_pd(hat_c + i * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + i * ldc), R00 + i));
    //}
    _mm256_storeu_pd(hat_c + 0 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 0 * ldc), R00));
    _mm256_storeu_pd(hat_c + 1 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 0 * ldc), R01));
    _mm256_storeu_pd(hat_c + 2 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 0 * ldc), R02));
    _mm256_storeu_pd(hat_c + 3 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 0 * ldc), R03));
    _mm256_storeu_pd(hat_c + 4 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 0 * ldc), R04));

    _mm256_storeu_pd(hat_c + 5 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 0 * ldc), R05));
    _mm256_storeu_pd(hat_c + 6 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 0 * ldc), R06));
    _mm256_storeu_pd(hat_c + 7 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 0 * ldc), R07));
    _mm256_storeu_pd(hat_c + 8 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 0 * ldc), R08));
    _mm256_storeu_pd(hat_c + 9 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 0 * ldc), R09));

    _mm256_storeu_pd(hat_c + 10 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 0 * ldc), R10));
    _mm256_storeu_pd(hat_c + 11 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 0 * ldc), R11));
}


inline void pack_b( double* __restrict__ src_b, double* __restrict__ pak_b, int ldb, int n ){
    for ( int row_i = 0; row_i < k_b; ++row_i ){
        double* src_b_row_i = src_b + row_i * ldb;
        double* pak_b_row_i = pak_b + row_i * n_r;

        UNROLL_LOOP( 4 )
        for ( int n_r_i = 0; n_r_i < (n / n_r); ++n_r_i ){
            double* src_b_row_n_r_i = src_b_row_i + n_r_i * n_r;
            double* pak_b_row_n_r_i = pak_b_row_i + n_r_i * k_b * n_r;

            UNROLL_LOOP( n_r )
            for ( int col_i = 0; col_i < n_r; ++col_i ){
                *(pak_b_row_n_r_i + col_i) = *(src_b_row_n_r_i + col_i);
            }
        }
    }
}


inline void pack_a( double* __restrict__ src_a, double* __restrict__ pak_a, int lda ){
    for ( int m_r_i = 0; m_r_i < (m_b / m_r); ++m_r_i ){
        double* src_a_row_m_r_i = src_a + m_r_i * m_r * lda;
        double* pak_a_row_m_r_i = pak_a + m_r_i * m_r * k_b;

        UNROLL_LOOP( 4 )
        for ( int row_i = 0; row_i < m_r; ++row_i ){
            double* src_a_row_i = src_a_row_m_r_i + row_i * lda;
            double* pak_a_row_i = pak_a_row_m_r_i + row_i;

            UNROLL_LOOP( 8 * 4 )
            for ( int col_i = 0; col_i < k_b; ++col_i ){
                *(pak_a_row_i + col_i * m_r) = *(src_a_row_i + col_i);
            }
        }
    }
}



void dgemm_tigerlake( int m, int k, int n, \
                double* src_a, double* src_b, double* src_c, \
                int lda, int ldb, int ldc )
{
    // Memory for \tilde a and \tilde b
    double* pak_a = (double*)_mm_malloc( m_b * k_b * sizeof( double ), 64 );
    double* pak_b = (double*)_mm_malloc( k_b * n   * sizeof( double ), 64 );

    for ( int k_b_i = 0; k_b_i < k / k_b; k_b_i++){
        pack_b( src_b + k_b_i * k_b * ldb, pak_b, ldb, n );
        for ( int m_b_i = 0; m_b_i < m / m_b; m_b_i++ ){
            pack_a( src_a + m_b_i * m_b * lda + k_b_i * k_b, pak_a, lda );
            for ( int n_r_i = 0; n_r_i < n / n_r; n_r_i++ ){
                for ( int m_r_i = 0; m_r_i < m_b / m_r; m_r_i++ ){
                    inner_kernel( pak_a + m_r_i * m_r * k_b, \
                                  pak_b + n_r_i * n_r * k_b, \
                                  src_c + m_b_i * m_b * ldc + m_r_i * m_r * ldc + n_r_i * n_r, \
                                  ldc );
                }
            }
        }
    }

    _mm_free( pak_a );
    _mm_free( pak_b );
}


void initialize_matrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)i;//(double)(rand() % 100) / 10.0; // Random values between 0.0 and 10.0
    }
}

int main(){
    int M,N,K,l;
    l=2048;
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
    printf(" %f, %f, %f \n",C[0],C[1],B[1]);
    return 0;
}
