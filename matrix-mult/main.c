#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cblas.h>

#define N 1024
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
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
}

// 2. Reordered (ikj order)
void matmul_reordered(float* A, float* B, float* C) {
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
}

// 3. Blocked (tiled) matrix multiplication
void matmul_blocked(float* A, float* B, float* C) {
    const int BLOCK = BLOCK_SIZE;

    for (int i0 = 0; i0 < N; i0 += BLOCK) {
        for (int j0 = 0; j0 < N; j0 += BLOCK) {
            for (int k0 = 0; k0 < N; k0 += BLOCK) {
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

double benchmark(void (*func)(float*, float*, float*)) {

    float *A = aligned_alloc(64, N*N*sizeof(float));
    float *B = aligned_alloc(64, N*N*sizeof(float));
    float *C = aligned_alloc(64, N*N*sizeof(float));
    
    for (int i = 0; i < N*N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }
    memset(C, 0, N*N*sizeof(float));
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    func(A, B, C);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time = (end.tv_sec - start.tv_sec) + 
                  (end.tv_nsec - start.tv_nsec) / 1e9;
    
    double gflops = (2.0 * N * N * N) / time / 1e9;
    
    printf("N=%4d: %8.3f ms, %6.2f GFLOPS\n", N, time*1000, gflops);
    
    free(A); free(B); free(C);
    return gflops;
}

int main() {

    printf("[BLAS]\n");
    benchmark(matmul_blas);
    
    // printf("[Naive]\n");
    // benchmark(matmul_naive);

    printf("[Naive Reordered]\n");
    benchmark(matmul_reordered);

    // int blocks[] = {16, 24, 32, 40, 48, 56, 64, 96};
    // int length = sizeof(blocks) / sizeof(blocks[0]);
    // for(int i = 0; i < length; i++) {
    //     BLOCK_SIZE = blocks[i];
    //     printf("[Blocked: BLOCK_SIZE=%d]\n", BLOCK_SIZE);
    //     benchmark(matmul_blocked);
    // }

    // BLOCK_SIZE = 48 should be close to optimal
    BLOCK_SIZE = 48;
    printf("[Blocked: BLOCK_SIZE=%d]\n", BLOCK_SIZE);
    benchmark(matmul_blocked);
    

    return 0;
}
