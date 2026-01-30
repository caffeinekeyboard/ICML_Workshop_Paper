import torch
import torch.nn as nn
import torch.fft as fft

def dct2(x, norm="ortho"):
    """
    Compute the Discrete Cosine Transform (DCT) Type II of the input tensor.
    
    Args:
        x (torch.Tensor): Input tensor of shape (..., N).
        norm (str, optional): Normalization mode. Default is "ortho".
        
    Returns:
        torch.Tensor: DCT Type II transformed tensor of the same shape as input.
        
    This function computes the DCT Type II by leveraging the Fast Fourier Transform (FFT) using the algorithm described by J. Makhoul in 1980.
    For more information on the theory behind this implementation, refer to: https://doi.org/10.1109/TASSP.1980.1163351
    """
    pass