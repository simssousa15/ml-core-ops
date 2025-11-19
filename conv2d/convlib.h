#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

// copy pasted methods from previous implementations
// will find a better solution later

void im2col(
    const float *input,
    int C, int H, int W,
    int K_h, int K_w,
    int pad, int stride,
    float *output
);

// 4.2. AVX2 SIMD version with FMA, unrolled loop and accumulators
// "My guess is the performance caps at around ~8-12 accumulators" - chatgpt
void matmul_avx2_acc(float* A, float* B, float* C, int N);

void conv2d(const float* input, const float* kernel,
            float* output,
            int C_in, int H_in, int W_in,
            int C_out,
            int K_h, int K_w,
            int stride, int pad);