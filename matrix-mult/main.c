#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cblas.h>
#include <immintrin.h>
#include <math.h>

#define N 2048  // Matrix size N x N
static int BLOCK_SIZE;

// 0. Using BLAS library
// Peak performance reference
void matmul_blas(float* A, float* B, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,           // M, N, K dimensions
                1.0f,              // alpha scalar
                A, N,              // matrix A, leading dimension
                B, N,              // matrix B, leading dimension
                0.0f,              // beta scalar (0 = overwrite C)
                C, N);             // matrix C, leading dimension
}

// 1. Naive (ijk order)
void matmul_naive(float* A, float* B, float* C) {
    memset(C, 0, N*N*sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
}

// 2. Reordered (ikj order)
void matmul_reordered(float* A, float* B, float* C) {
    memset(C, 0, N*N*sizeof(float));
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
}

// 3. Blocked (tiled) matrix multiplication
void matmul_blocked(float* A, float* B, float* C) {
    memset(C, 0, N*N*sizeof(float));
    const int BLOCK = BLOCK_SIZE;

    for (int i0 = 0; i0 < N; i0 += BLOCK) {
        for (int k0 = 0; k0 < N; k0 += BLOCK) {
            for (int j0 = 0; j0 < N; j0 += BLOCK) {
                // Process BLOCK×BLOCK sub-matrix
                int i_max = (i0 + BLOCK < N) ? i0 + BLOCK : N;
                int j_max = (j0 + BLOCK < N) ? j0 + BLOCK : N;
                int k_max = (k0 + BLOCK < N) ? k0 + BLOCK : N;
                
                for (int i = i0; i < i_max; i++) {
                    for (int k = k0; k < k_max; k++) {
                        float a_ik = A[i*N + k];
                        for (int j = j0; j < j_max; j++) {
                            C[i*N + j] += a_ik * B[k*N + j];
                        }
                    }
                }
            }
        }
    }
}

//4. AVX2 SIMD version with FMA
void matmul_avx2(float* A, float* B, float* C) {
    memset(C, 0, N*N*sizeof(float));
    const int BLOCK = BLOCK_SIZE;

    for (int i0 = 0; i0 < N; i0 += BLOCK) {
        for (int k0 = 0; k0 < N; k0 += BLOCK) {
            for (int j0 = 0; j0 < N; j0 += BLOCK) {
                // Process BLOCK×BLOCK sub-matrix
                int i_max = (i0 + BLOCK < N) ? i0 + BLOCK : N;
                int j_max = (j0 + BLOCK < N) ? j0 + BLOCK : N;
                int k_max = (k0 + BLOCK < N) ? k0 + BLOCK : N;
                
                 // Compute block C[i0:i_max][j0:j_max]
                for (int i = i0; i < i_max; i++) {
                    for (int k = k0; k < k_max; k++) {

                        // Broadcast A[i][k] to AVX2 register
                        // float a_ik = A[i*N + k];
                        __m256 a_ik = _mm256_set1_ps(A[i*N + k]);
                        
                        int j = j0;
                        // Vectorized loop: 8 floats at a time
                        for (; j + 8 <= j_max; j += 8) {
                            __m256 c_vec = _mm256_loadu_ps(&C[i*N + j]);   // load C[i][j:j+7]
                            __m256 b_vec = _mm256_loadu_ps(&B[k*N + j]);   // load B[k][j:j+7]
                            c_vec = _mm256_fmadd_ps(a_ik, b_vec, c_vec);   // C += A*B
                            _mm256_storeu_ps(&C[i*N + j], c_vec);          // store back
                        }
                    }
                }
            }
        }
    }
}


double benchmark(void (*func)(float*, float*, float*), float *A, float *B, float *C, float *res, char* name) {
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    func(A, B, C);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time = (end.tv_sec - start.tv_sec) + 
                  (end.tv_nsec - start.tv_nsec) / 1e9;
    
    double gflops = (2.0 * N * N * N) / time / 1e9;
    
    // Verify correctness
    float err = 0.0f;
    float max_err = 0.00f;
    for (int i = 0; i < N*N; i++) {
        err = fabs(C[i] - res[i]);
        if(err > max_err) {
            max_err = err;
        }
    }

    printf("[Perf] %-10s %6.2f GFLOPS | max_err: %.1e\n", name, gflops, max_err);
    return gflops;
}


void test_avx2_works() {
    float a_scalar = 2.0f;
    float b[2048] __attribute__((aligned(32)));
    float c_scalar[2048] __attribute__((aligned(32)));
    float c_avx2[2048] __attribute__((aligned(32)));
    
    for(int i = 0; i < 2048; i++) {
        b[i] = i;
        c_scalar[i] = 0;
        c_avx2[i] = 0;
    }
    
    // Scalar version
    clock_t start = clock();
    for(int rep = 0; rep < 2048; rep++) {
        for(int i = 0; i < 2048; i++) {
            c_scalar[i] += a_scalar * b[i];
        }
    }
    clock_t end = clock();
    double scalar_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    // AVX2 version
    start = clock();
    __m256 a_vec = _mm256_set1_ps(a_scalar);
    for(int rep = 0; rep < 2048; rep++) {
        for(int i = 0; i < 2048; i += 8) {
            __m256 b_vec = _mm256_load_ps(&b[i]);
            __m256 c_vec = _mm256_load_ps(&c_avx2[i]);
            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            _mm256_store_ps(&c_avx2[i], c_vec);
        }
    }
    end = clock();
    double avx2_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("\n[Microbenchmark]\n");
    printf("Scalar: %.3f s\n", scalar_time);
    printf("AVX2:   %.3f s\n", avx2_time);
    printf("Speedup: %.2fx\n", scalar_time / avx2_time);
    
    // Verify correctness
    int errors = 0;
    for(int i = 0; i < 2048; i++) {
        if(fabs(c_scalar[i] - c_avx2[i]) > 0.01) errors++;
    }
    printf("Errors: %d\n", errors);
    printf("==========================\n");
}

void init(float *A, float *B, float *C) {
    srand(42);
    for (int i = 0; i < N*N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }
    memset(C, 0, N*N*sizeof(float));
}

int main() {

    //test_avx2_works();

    float *A = aligned_alloc(64, N*N*sizeof(float));
    float *B = aligned_alloc(64, N*N*sizeof(float));
    float *C = aligned_alloc(64, N*N*sizeof(float));

    init(A, B, C);

    // True value to ensure correctness
    float *res = aligned_alloc(64, N*N*sizeof(float));
    memset(res, 0, N*N*sizeof(float));
    matmul_blas(A, B, res);


    //set single thread for BLAS
    openblas_set_num_threads(1);
    openblas_get_num_threads(); // force initialization

    benchmark(matmul_blas, A, B, C, res, "BLAS");
    // benchmark(matmul_naive, A, B, C, res, "NAIVE");
    benchmark(matmul_reordered, A, B, C, res, "REORDERED");

    // int blocks[] = {16, 24, 32, 40, 48, 56, 64, 96};
    // int length = sizeof(blocks) / sizeof(blocks[0]);
    // for(int i = 0; i < length; i++) {
    //     BLOCK_SIZE = blocks[i];
    //     benchmark(matmul_blocked, A, B, C, res, "BLOCKED");
    // }

    // BLOCK_SIZE = 48 should be close to optimal
    // N = 5096 -> 0.54GFLOPS(reordered) to 0.61GFLOPS 
    // Around 13% improvement using -O0
    BLOCK_SIZE = 32;
    benchmark(matmul_blocked, A, B, C, res, "BLOCKED");
    benchmark(matmul_avx2, A, B, C, res, "AVX2");
    
    free(A);free(B);free(C);free(res);
    return 0;
}
