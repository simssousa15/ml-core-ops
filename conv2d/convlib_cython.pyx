# convlib.pyx
from libc.stdlib cimport malloc, free

cdef extern from "convlib.h":
    void im2col(const float *input,
                int C, int H, int W,
                int K_h, int K_w,
                int pad, int stride,
                float *output)

    void matmul_avx2_acc(float* A, float* B, float* C, int N)

    void conv2d(const float* input, const float* kernel,
                float* output,
                int C_in, int H_in, int W_in,
                int C_out,
                int K_h, int K_w,
                int stride, int pad)

def py_im2col(float[::1] input, int C, int H, int W,
              int K_h, int K_w, int pad, int stride):
    cdef int out_h = (H + 2*pad - K_h)//stride + 1
    cdef int out_w = (W + 2*pad - K_w)//stride + 1
    cdef int out_size = C * K_h * K_w * out_h * out_w
    cdef float[::1] output = <float[:out_size]> malloc(out_size * sizeof(float))
    if not output.base:
        raise MemoryError()
    im2col(&input[0], C, H, W, K_h, K_w, pad, stride, &output[0])
    return output

def py_matmul_avx2_acc(float[::1] A, float[::1] B, int N):
    cdef float[::1] Cbuf = <float[:N*N]> malloc(N*N*sizeof(float))
    if not Cbuf.base:
        raise MemoryError()
    matmul_avx2_acc(&A[0], &B[0], &Cbuf[0], N)
    return Cbuf

def py_conv2d(float[::1] input, float[::1] kernel,
              int C_in, int H_in, int W_in,
              int C_out, int K_h, int K_w,
              int stride, int pad):
    cdef int H_out = (H_in + 2*pad - K_h)//stride + 1
    cdef int W_out = (W_in + 2*pad - K_w)//stride + 1
    cdef int out_size = C_out * H_out * W_out
    cdef float[::1] output = <float[:out_size]> malloc(out_size * sizeof(float))
    if not output.base:
        raise MemoryError()
    conv2d(&input[0], &kernel[0], &output[0],
           C_in, H_in, W_in, C_out,
           K_h, K_w, stride, pad)
    return output