import torch
import torch.nn as nn

class FeatureCorrelation2D(nn.Module):
    """
    Computes the dense feature correlation between two 2D feature maps.
    Migrated from the 3D GumNet architecture for 2D ICML Workshop Paper.
    
    References
    ----------
    [1] Convolutional neural network architecture for geometric matching, Ignacio Rocco, et al.
    [2] Gum-Net: Unsupervised Geometric Matching for Fast and Accurate 3D Subtomogram Image Alignment and Averaging, Zeng et al.
    """
    def __init__(self):
        super(FeatureCorrelation2D, self).__init__()

    def forward(self, f_A: torch.Tensor, f_B: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the 2D feature correlation.
        
        Args:
            f_A (torch.Tensor): Feature map A of shape [B, C, H_A, W_A]
            f_B (torch.Tensor): Feature map B of shape [B, C, H_B, W_B]
            
        Returns:
            torch.Tensor: Correlation map of shape [B, H_A * W_A, H_B, W_B]
        """
        B, C, H_A, W_A = f_A.size()
        B_b, C_b, H_B, W_B = f_B.size()
        assert B == B_b and C == C_b, "Batch size and Channel dimensions must match."
        f_A_flat = f_A.view(B, C, -1)
        f_A_transposed = f_A_flat.transpose(1, 2).contiguous()
        f_B_flat = f_B.view(B, C, -1)
        correlation_tensor = torch.bmm(f_A_transposed, f_B_flat)
        correlation_tensor = correlation_tensor.view(B, H_A * W_A, H_B, W_B)

        return correlation_tensor