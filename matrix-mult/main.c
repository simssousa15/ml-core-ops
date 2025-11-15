#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1024

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
    printf("[Naive]\n");
    benchmark(matmul_naive);

    printf("[Naive Reordered]\n");
    benchmark(matmul_reordered);

    return 0;
}
