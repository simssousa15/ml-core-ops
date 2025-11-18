#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

void error_check(float* output, float* expected, int size) {
    double max_err = 0.0;
    for (int i = 0; i < size; i++) {
        double err = fabs(output[i] - expected[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    printf("Max error: %.4f\n", max_err);
}

// NOTE: Better to move the testing to python for ease of use

int main(){

    // ### Set sizes ###
    // img: C x H x W
    const int C = 3;
    const int H = 64;
    const int W = 64;

    // kernel: KH x KW and stride and padding
    const int KH = 3;
    const int KW = 3;
    const int stride = 1;
    const int padding = 1;

    // output col size
    const int H_out = (H + 2*padding - KH) / stride + 1;
    const int W_out = (W + 2*padding - KW) / stride + 1;
    const int col_len = C * KH * KW * H_out * W_out;
    // ##################

    // print sizes
    printf("Image size: C=%d, H=%d, W=%d\n", C, H, W);
    printf("Kernel size: KH=%d, KW=%d, stride=%d, padding=%d\n", KH, KW, stride, padding);
    printf("Output col size: %d x %d = %d\n", C*KH*KW, H_out*W_out, col_len);

    float *img = aligned_alloc(64, C*H*W*sizeof(float));
    //load image from data/img.bin
    FILE *f = fopen("data/img.bin", "rb");
    fread(img, sizeof(float), C*H*W, f);
    fclose(f);

    float* c_col = aligned_alloc(64, col_len * sizeof(float));
    im2col(img, C, H, W, KH, KW, padding, stride, c_col);

    float* torch_col = malloc(col_len * sizeof(float));
    // load expected output from data/col.bin
    FILE *f2 = fopen("data/col.bin", "rb");
    fread(torch_col, sizeof(float), col_len, f2);
    fclose(f2);

    // Check for correctness
    error_check(c_col, torch_col, col_len);

    free(img);   free(c_col);   free(torch_col);
    return 0;
}