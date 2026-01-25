from typing import List


def convolution_out_size(in_size, kernel_size, stride, padding):
    return int((in_size - kernel_size + 2 * padding) / stride + 1)

def pooling_out_size(in_size, kernel_size, stride):
    return int((in_size - kernel_size) / stride + 1)

def cnn_out_size(
    num_layers: int, 
    in_size: int, 
    kernels_size_conv: List,
    stride_conv: List,
    padding_conv: List,
    kernel_size_pool: List,
    stride_pool: List
    ):
    
    """
    Calculate the output size of a Convoluational Neural Network (CNN).
    """
    
    for i in range(num_layers - 1):
        out_size = convolution_out_size(in_size, kernels_size_conv[i], stride_conv[i], padding_conv[i])
        if kernel_size_pool is not None and stride_pool is not None:
            out_size = pooling_out_size(out_size, kernel_size_pool[i], stride_pool[i])
        in_size = out_size
    out_size = convolution_out_size(in_size, kernels_size_conv[-1], stride_conv[-1], padding_conv[-1])
    return out_size