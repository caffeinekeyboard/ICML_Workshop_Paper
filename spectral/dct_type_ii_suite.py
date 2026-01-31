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
    x_shape = x.shape
    signal_length = x.shape[-1]
    signal_list = x.reshape(-1, signal_length)
    markhoul_shuffled_signal_list = torch.cat([signal_list[:, ::2], signal_list[:, 1::2].flip([1])], dim=1)
    fourier_transformed_signal_list = fft.fft(markhoul_shuffled_signal_list, dim=1)
    twiddle_factor_indices = - torch.arange(signal_length, dtype = x.dtype, device=x.device).unsqueeze(0) * torch.pi / (2 * signal_length)
    twiddle_factor = torch.exp(1j * twiddle_factor_indices)
    final_signal_list = (fourier_transformed_signal_list * twiddle_factor).real
    
    if norm == "ortho":
        final_signal_list[:, 0] /= (2 * signal_length ** 0.5)
        final_signal_list[:, 1:] /= (2 * ((signal_length / 2)**0.5))
        
    output_signals = 2 * final_signal_list.view(*x_shape)
    return output_signals

def idct2():
    pass