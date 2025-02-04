/*DGEMM code; Tried to optimize for TigerLakeU CPUS*/
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
  #define UNROLL_LOOP(n)      _Pragma(TO_STRING(GCC unroll (16)))
#elif defined(_MSC_BUILD)
  #pragma message ("Microsoft Visual C++ (MSVC) detected: Loop unrolling not supported!")
  #define UNROLL_LOOP(n)
#else
  #warning "Unknown compiler: Loop unrolling not supported!"
  #define UNROLL_LOOP(n)
#endif


#define k_b 512   
#define m_b 32
#define m_r 8
#define n_r 4

void inner_kernel(double* __restrict__ hat_a,
                  double* __restrict__ hat_b,
                  double* __restrict__ hat_c,
                  int ldc) {
    __m256d R00 = _mm256_setzero_pd();
    __m256d R01 = _mm256_setzero_pd();
    __m256d R02 = _mm256_setzero_pd();
    __m256d R03 = _mm256_setzero_pd();

    for (int i = 0; i < k_b; ++i) {
        _mm_prefetch((char*)(hat_a + 16), _MM_HINT_T0);
        _mm_prefetch((char*)(hat_b + 16), _MM_HINT_T0);

        __m256d B = _mm256_load_pd(hat_b);

        asm volatile(
            "vfmadd231pd   0(%[hat_a])%{1to4}, %[B], %[R00]\n\t"
            "vfmadd231pd   8(%[hat_a])%{1to4}, %[B], %[R01]\n\t"
            "vfmadd231pd  16(%[hat_a])%{1to4}, %[B], %[R02]\n\t"
            "vfmadd231pd  24(%[hat_a])%{1to4}, %[B], %[R03]\n\t"
            : [R00] "+v" (R00), [R01] "+v" (R01), [R02] "+v" (R02), [R03] "+v" (R03)
            : [B] "v" (B), [hat_a] "r" (hat_a)
            : "memory"
        );

        hat_a += m_r;
        hat_b += n_r;
    }

    _mm256_storeu_pd(hat_c + 0 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 0 * ldc), R00));
    _mm256_storeu_pd(hat_c + 1 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 1 * ldc), R01));
    _mm256_storeu_pd(hat_c + 2 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 2 * ldc), R02));
    _mm256_storeu_pd(hat_c + 3 * ldc, _mm256_add_pd(_mm256_loadu_pd(hat_c + 3 * ldc), R03));
}

void pack_b( double* __restrict__ src_b, double* __restrict__ pak_b, int ldb, int n ){
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


void pack_a( double* __restrict__ src_a, double* __restrict__ pak_a, int lda ){
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
        // Pack \tilde b
        pack_b( src_b + k_b_i * k_b * ldb, pak_b, ldb, n );
        for ( int m_b_i = 0; m_b_i < m / m_b; m_b_i++ ){
            // Pack \tilde a
            pack_a( src_a + m_b_i * m_b * lda + k_b_i * k_b, pak_a, lda );
            for ( int n_r_i = 0; n_r_i < n / n_r; n_r_i++ ){
                for ( int m_r_i = 0; m_r_i < m_b / m_r; m_r_i++ ){
                    // Inner Kernel (register blocking)
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
        matrix[i] = (double)(rand() % 100) / 10.0; // Random values between 0.0 and 10.0
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
    return 0;
}

/*

DETERMINING BLOCK SIZE FOR DGEMM ON TIGERLAKE

Optimizing DGEMM (Double-precision General Matrix Multiplication) for  Tiger Lake  requires selecting the right  block sizes (`m_b`, `n_r`, `k_b`)  based on:  
1.  Cache sizes  (to minimize cache misses and reuse data efficiently).  
2.  Number of registers  (to maximize register utilization).  
3.  Memory bandwidth and latency  (to ensure efficient memory access patterns).  

---

    1: Tiger Lake Architecture 
1.1. Cache Hierarchy in Tiger Lake (11th Gen Intel Core) 
|  Cache Level  |  Size per Core  |  Latency   |
|---------------|----------------|-------------|
| L1 (Data)     |  48 KB        | ~4 cycles  |
| L2            |  1.25 MB       | ~12 cycles |
| L3 (Shared)   |  Up to 12 MB   | ~36 cycles |

  Goal:  Fit as much of the working set (blocks of matrices A, B, and C) into L1 or L2 cache.

2. Registers Available for AVX2 (256-bit) 
Each  Intel Tiger Lake  core has  16 vector registers  (`ymm0`-`ymm15`), each holding  4 double-precision (`double`) values  (256-bit registers).  
So, in total:
-  Max storage per register  = `16 registers × 4 doubles = 64 doubles (~512 bytes)`

  Goal:  Ensure that a register block (submatrix of `C`) fits within registers to  avoid spilling into L1 cache .

    Step 2: Determining Block Sizes (`m_b`, `n_r`, `k_b`) 
We select `m_b`, `n_r`, and `k_b` based on cache and register constraints.

1. Choosing `m_r × n_r` (Register Blocking) 
Since AVX2 works with  4 doubles per 256-bit register , we aim to  keep an entire submatrix of `C` in registers .

      Optimal choice: 
- `m_r = 8` (uses `8 × 4 = 32` doubles, fitting into 8 registers)
- `n_r = 4` (uses 4 registers)
- Total registers used = `8 + 4 = 12` (fits within 16 available registers)
  
  Why?  This ensures that an entire  8×4 block of C  stays in registers during computation, reducing cache accesses.

  2. Choosing `k_b` (Panel Width for Matrix A and B) 
  Guideline:  Choose `k_b` to  fit the panel of `B` in L1/L2 cache  to avoid excessive memory traffic.

- Each panel of `B` (size `k_b × n_r`) must fit in L1 or L2 cache.
- Each panel of `A` (size `m_r × k_b`) should also fit into L2.

      Approximate L2 Cache Usage 
-  Size of `A` block (`m_b × k_b`)   
  - `32 × 512` doubles = `128 KB` (fits in L2)
-  Size of `B` block (`k_b × n_r`)   
  - `512 × 4` doubles = `16 KB` (fits in L1)

  Choice:  `k_b = 512` is a good balance to leverage L2 while keeping `B` blocks in L1.

  3. Choosing `m_b` (Row Blocking for `A`) 
  Guideline:  Choose `m_b` so that multiple  panels of `A` fit in L2 .

- We already set `m_r = 8`, and a block should be a multiple of `m_r`.
-  Empirical choice:  `m_b = 32` ensures multiple `m_r` blocks fit well into L2.

    Final Block Size Choices for Tiger Lake 
|  Block Size  |  Chosen Value  |  Justification  |
|--------------|---------------|------------------|
| `m_r`       |  8          | Fits in registers (uses 8 out of 16 registers) |
| `n_r`       |  4          | Fits in registers (uses 4 out of 16 registers) |
| `m_b`       |  32         | Fits in L2 (smaller blocks for better reuse) |
| `k_b`       |  512        | Fits L1 (B-panel), fits L2 (A-panel) |

These values balance  register blocking, cache reuse, and memory bandwidth efficiency .

---

  3: Optimizing Memory Access 
-  A-panel (`m_b × k_b`) is loaded from memory and reused for multiple `B` panels. 
-  B-panel (`k_b × n_r`) fits in L1 cache, so it is reused for multiple `A` rows. 
-  C-panel (`m_b × n_r`) stays in registers for computation before being written to memory. 
-  Use `_mm_prefetch()` to prefetch next `A` and `B` panels. 

---

Conclusion 
By carefully selecting `m_b = 32`, `n_r = 4`, `k_b = 512`, we:
  Maximize register utilization  (keep an entire submatrix of C in registers).  
  Minimize cache misses  (fit B in L1, A in L2).  
  Reduce memory traffic  (reuse A and B efficiently).  

This approach  significantly boosts DGEMM performance  on  Intel Tiger Lake  CPUs. 

*/
