import torch
from torch.nn import Unfold
import numpy as np
import os

# source pytorch/bin/activate
if __name__ == "__main__":
    # Generate a ranfom image (C=3, H=64, W=64)
    img = torch.randn(1, 3, 64, 64)
    
    # set kernel_size=3, padding=1, stride=2
    unfold = Unfold(
        kernel_size=(3, 3),
        padding=1,
        stride=1
    )

    # convert to im2col
    cols = unfold(img)

    os.makedirs("data", exist_ok=True)
    img.numpy()[0].astype(np.float32).tofile("data/img.bin")
    cols.numpy()[0].astype(np.float32).tofile("data/col.bin")

    # Print sizes
    print("Image shape:", img.numpy()[0].shape)
    print("Col shape:", cols.numpy()[0].shape)