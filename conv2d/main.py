import convlib
import numpy as np
import torch
import os
import time

# Force CPU only: ensure no CUDA is used
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Force PyTorch to single-thread CPU execution
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def time_it(fn, *args, repeats=50):
    start = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    end = time.perf_counter()
    return (end - start) / repeats


if __name__ == "__main__":
    # ----- Parameters -----
    C_in, H, W = 3, 8, 8
    K_h = K_w = 3
    C_out = 4
    s, p = 1, 1

    reps = 200

    # ----- Input generation -----
    input_arr = np.random.randn(C_in * H * W).astype(np.float32)
    kernel_arr = np.random.randn(C_out * C_in * K_h * K_w).astype(np.float32)

    # ----- Your custom conv (numpy-based input) -----
    out_custom = convlib.py_conv2d(
        input_arr,
        kernel_arr,
        C_in, H, W,
        C_out,
        K_h, K_w,
        stride=s, pad=1
    )

    # ----- Prepare tensors for PyTorch -----
    # reshape to NCHW
    x_torch = torch.tensor(input_arr.reshape(1, C_in, H, W))
    k_torch = torch.tensor(kernel_arr.reshape(C_out, C_in, K_h, K_w))

    # ----- PyTorch conv -----
    out_torch = torch.nn.functional.conv2d(
        x_torch,
        k_torch,
        stride=s,
        padding=p
    )

    # ----- Compare outputs -----
    out_torch_np = out_torch.detach().cpu().numpy().ravel()

    print("Max difference:", np.abs(out_custom - out_torch_np).max())

    # ----- Benchmark -----
    t_custom = time_it(
        lambda: convlib.py_conv2d(
            input_arr,
            kernel_arr,
            C_in, H, W,
            C_out,
            K_h, K_w,
            stride=s, pad=1
        ),
        repeats=reps
    )

    t_torch = time_it(
        lambda: torch.nn.functional.conv2d(
            x_torch,
            k_torch,
            stride=s,
            padding=p
        ),
        repeats=reps
    )

    print(f"Custom conv avg time: {t_custom * 1e6:.2f} µs")
    print(f"PyTorch conv avg time: {t_torch * 1e6:.2f} µs")