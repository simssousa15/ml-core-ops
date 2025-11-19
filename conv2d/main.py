import sys
sys.path.insert(0, "build")
import convlib

import numpy as np


# Build library
# python setup.py build_ext --build-lib build --build-temp build/temp

# # Example conv input
C_in, H, W = 3, 8, 8
K_h = K_w = 3
C_out = 4

input_arr = np.random.randn(C_in * H * W).astype(np.float32)
kernel_arr = np.random.randn(C_out * C_in * K_h * K_w).astype(np.float32)

out = convlib.py_conv2d(
    input_arr,
    kernel_arr,
    C_in, H, W,
    C_out,
    K_h, K_w,
    stride=1, pad=1
)

print(np.array(out))