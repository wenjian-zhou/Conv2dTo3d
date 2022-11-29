import torch
from torch import nn
import math 

"""
X: input features
K: convolution kernel
BitsSize: size of the bit string (along z-axis) at a position on the 2d occupancy map
"""
def corr2dTo3d(X, K, BitsSize, stride, padding):
    """compute 2d (with bits as z-axis) to 3d using cross-correlation"""
    d, h, w = K.shape
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding

    """padding operation"""
    augmented_h = X.shape[0] + 2 * pad_h 
    augmented_w = X.shape[1] + 2 * pad_w 

    augmented_X = torch.zeros(augmented_h, augmented_w).to(torch.int)
    augmented_X[pad_h : pad_h + X.shape[0], pad_w : pad_w + X.shape[1]] = X

    """final 3d tensor after convolution"""
    Y = torch.zeros(math.floor((BitsSize + 2 * pad_d - (d - 1) - 1)/stride_d + 1), math.floor((X.shape[0] + 2 * pad_h - (h - 1) - 1)/stride_h + 1), math.floor((X.shape[1] + 2 * pad_w - (w - 1) - 1)/stride_w + 1))

    for k in range(Y.shape[0]):
        for i in range(Y.shape[1]):
            for j in range(Y.shape[2]):
                for step in range(d):
                    """treat every convolution kernel as slices, for each slice we can extract the corresponding bit"""
                    Y[k, i, j] += ((((augmented_X[i * stride_h : i * stride_h + h, j * stride_w : j * stride_w + w]) << pad_d >> (k * stride_d + step)) % 2).to(torch.float32) * K[step, 0 : h,0 : w]).sum()

    return Y

class Conv2dTo3d(nn.Module):
    def __init__(self, kernel_size, bits_size, stride = (1, 1, 1), padding = (0, 0, 0)):
        super().__init__()
        """random weight between [0,1], and 0 bias"""
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
        self.bits_size = bits_size
        self.stride = stride 
        self.padding = padding
    
    def forward(self, x):
        return corr2dTo3d(x, self.weight, self.bits_size, self.stride, self.padding) + self.bias