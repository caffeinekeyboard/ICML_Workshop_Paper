import torch
import torch.nn as nn
import torch.fft as fft


def dct2(x, norm="ortho"):
    """
    Compute the Discrete Cosine Transform (DCT) Type II of the input tensor along its last dimension.
    
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
        final_signal_list[:, 0] /= 2.0 * signal_length ** 0.5
        final_signal_list[:, 1:] /= 2.0 * (signal_length / 2.0)**0.5
        
    output_signals = 2 * final_signal_list.view(*x_shape)
    return output_signals


def idct2(x, norm="ortho"):
    """
    Compute the Inverse Discrete Cosine Transform (IDCT) for DCT-II of the input tensor along its last dimension.
    
    Args:
        x (torch.Tensor): Input tensor of shape (..., N).
        norm (str, optional): Normalization mode. Default is "ortho".
    
    Returns:
        torch.Tensor: Inverse DCT Type II transformed tensor of the same shape as input.
        
    It is important to note that the inverse DCT Type II is equivalent to a scaled DCT Type III.
    For more information on the theory behind this implementation, refer to: https://doi.org/10.1109/TASSP.1980.1163351
    """
    x_shape = x.shape
    signal_length = x.shape[-1]
    signal_list = x.reshape(-1, signal_length) / 2.0
    
    if norm == "ortho":
        signal_list[:, 0] *= 2.0 * signal_length ** 0.5
        signal_list[:, 1:] *= 2.0 * (signal_length / 2.0)**0.5
    
    twiddle_factor_indices = torch.arange(signal_length, dtype = x.dtype, device=x.device).unsqueeze(0) * torch.pi / (2 * signal_length)
    twiddle_factor = torch.exp(1j * twiddle_factor_indices)
    signal_list_imag = torch.cat([torch.zeros_like(signal_list[:, :1]), -signal_list[:, 1:].flip([1])], dim=1)
    signal_list_complex = signal_list + 1j * signal_list_imag
    final_signal_list_complex = signal_list_complex * twiddle_factor
    output_signals_shuffled = fft.ifft(final_signal_list_complex, dim=1).real
    output_signal = torch.zeros_like(output_signals_shuffled)
    even_signal_length = (signal_length + 1) // 2
    output_signal[:, ::2] = output_signals_shuffled[:, :even_signal_length]
    output_signal[:, 1::2] = output_signals_shuffled[:, even_signal_length:].flip([1])
    output_signal = output_signal.view(*x_shape)
    return output_signal


def dct2_2d(x, norm=None):
    """
    Compute the Discrete Cosine Transform (DCT) Type II of a two-dimensional input tensor.
    
    Args:
        x (torch.Tensor): Input tensor of shape (..., N).
        norm (str, optional): Normalization mode. Default is "ortho".
        
    Returns:
        torch.Tensor: Inverse DCT Type II transformed two-dimensional tensor of the same shape as input.
    """
    X1 = dct2(x, norm=norm)
    X2 = dct2(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)




def idct_2d(X, norm=None):
    """
    Compute the Inverse Discrete Cosine Transform (IDCT) for DCT-II of a two-dimensional input tensor.
    
    Args:
        x (torch.Tensor): Input tensor of shape (..., N).
        norm (str, optional): Normalization mode. Default is "ortho".
        
    Returns:
        torch.Tensor: Inverse DCT Type II transformed two-dimensional tensor of the same shape as input.
    """
    x1 = idct2(X, norm=norm)
    x2 = idct2(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)




class LinearDCT(nn.linear):
    """
    Docstring for LinearDCT
    """
    pass