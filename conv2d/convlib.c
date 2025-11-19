#include "convlib.h"
#include <cblas.h>

// copy pasted methods from previous implementations
// will find a better solution later

void im2col(
    const float *input,
    int C, int H, int W,
    int K_h, int K_w,
    int pad, int stride,
    float *output
) {
    int H_out = (H + 2 * pad - K_h) / stride + 1;
    int W_out = (W + 2 * pad - K_w) / stride + 1;

    int col = 0;

    for (int h_out = 0; h_out < H_out; ++h_out) {
        for (int w_out = 0; w_out < W_out; ++w_out) {

            int col_index = col++;

            for (int c = 0; c < C; ++c) {
                for (int kh = 0; kh < K_h; ++kh) {
                    for (int kw = 0; kw < K_w; ++kw) {

                        int h_in = h_out * stride + kh - pad;
                        int w_in = w_out * stride + kw - pad;

                        int row = c*K_h*K_w + kh*K_w + kw;

                        if (h_in < 0 || h_in >= H ||
                            w_in < 0 || w_in >= W) {
                            output[row * (H_out*W_out) + col_index] = 0.0f;
                        } else {
                            output[row * (H_out*W_out) + col_index] =
                                input[c*H*W + h_in*W + w_in];
                        }
                    }
                }
            }
        }
    }
}

// 4.2. AVX2 SIMD version with FMA, unrolled loop and accumulators
// "My guess is the performance caps at around ~8-12 accumulators" - chatgpt
void matmul_avx2_acc(float* A, float* B, float* C, int N) {
    memset(C, 0, N*N*sizeof(float));
    const int BLOCK = 64;  // outer L1/L2 block
    const int MICRO = 8;   // micro-kernel size

    for (int i0 = 0; i0 < N; i0 += BLOCK) {
        for (int j0 = 0; j0 < N; j0 += BLOCK) {
            for (int k0 = 0; k0 < N; k0 += BLOCK) {

                int i_max = (i0 + BLOCK < N) ? i0 + BLOCK : N;
                int j_max = (j0 + BLOCK < N) ? j0 + BLOCK : N;
                int k_max = (k0 + BLOCK < N) ? k0 + BLOCK : N;

                for (int i = i0; i < i_max; i++) {
                    for (int k = k0; k+8 <= k_max; k+=8) {
                        
                        //new kernel
                        __m256 c_m[8];
                        __m256 a_m[8];

                        // Load C[i0:i0+8, j0:j0+8] into registers
                        for (int _ = 0; _ < 8; _++){
                            c_m[_] = _mm256_load_ps(&C[(i*N + j0 + _*8)]);
                            a_m[_] = _mm256_set1_ps(A[i*N + k + _]);
                        }
                            

                        for (size_t _k = 0; _k < 8; _k++)
                        {
                            c_m[0] = _mm256_fmadd_ps(a_m[_k], _mm256_load_ps(&B[(k+_k)*N + j0 + 0]), c_m[0]);
                            c_m[1] = _mm256_fmadd_ps(a_m[_k], _mm256_load_ps(&B[(k+_k)*N + j0 + 8]), c_m[1]);
                            c_m[2] = _mm256_fmadd_ps(a_m[_k], _mm256_load_ps(&B[(k+_k)*N + j0 + 16]), c_m[2]);
                            c_m[3] = _mm256_fmadd_ps(a_m[_k], _mm256_load_ps(&B[(k+_k)*N + j0 + 24]), c_m[3]);
                            c_m[4] = _mm256_fmadd_ps(a_m[_k], _mm256_load_ps(&B[(k+_k)*N + j0 + 32]), c_m[4]);
                            c_m[5] = _mm256_fmadd_ps(a_m[_k], _mm256_load_ps(&B[(k+_k)*N + j0 + 40]), c_m[5]);
                            c_m[6] = _mm256_fmadd_ps(a_m[_k], _mm256_load_ps(&B[(k+_k)*N + j0 + 48]), c_m[6]);
                            c_m[7] = _mm256_fmadd_ps(a_m[_k], _mm256_load_ps(&B[(k+_k)*N + j0 + 56]), c_m[7]);
                        }
                        

                        // Store results back
                        for (int _ = 0; _ < 8; _++)
                            _mm256_store_ps(&C[(i*N + j0 + _*8)], c_m[_]);
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------
// conv2d using im2col + CBLAS sgemm
// -----------------------------------------------------
void conv2d(const float* input,
            const float* kernel,
            float* output,
            int C_in, int H_in, int W_in,
            int C_out,
            int K_h,  int K_w,
            int stride, int pad)
{
    int H_out = (H_in + 2*pad - K_h) / stride + 1;
    int W_out = (W_in + 2*pad - K_w) / stride + 1;

    int M = C_out;                  // rows of output (out_channels)
    int K = C_in * K_h * K_w;       // inner dimension
    int N = H_out * W_out;          // columns per output feature map

    // Allocate im2col buffer
    float* col = aligned_alloc(64, K * N * sizeof(float));

    // Compute im2col matrix
    im2col(input, C_in, H_in, W_in,
           K_h, K_w, pad, stride,
           col);

    // Use BLAS SGEMM:
    //
    // C = A * B
    //
    // A = kernel        (M × K)
    // B = col           (K × N)
    // C = output        (M × N)
    //
    // Row-major call:
    //
    // CBLAS_ORDER RowMajor
    // C = alpha*A*B + beta*C

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,     // A not transposed
        CblasNoTrans,     // B not transposed
        M,                // rows of A and C
        N,                // columns of B and C
        K,                // shared dimension
        1.0f,
        kernel, K,        // A pointer, lda = K
        col,    N,        // B pointer, ldb = N
        0.0f,
        output, N         // C pointer, ldc = N
    );

    free(col);
}


int main(){
    return 0;
}