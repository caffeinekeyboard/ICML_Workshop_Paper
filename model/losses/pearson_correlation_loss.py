import torch
import torch.nn as nn

class PearsonCorrelationLoss(nn.Module):
    """
    Pearson's Correlation Coefficient Loss (1 - r formulation).
    The 1-r^2 formulation used in the original paper is not well suited to the task of impression to template matching.
    """
    def __init__(self, eps=1e-8):
        super(PearsonCorrelationLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Calculates the 1 - r loss.
        
        Args:
            y_pred (torch.Tensor): Warped source feature map. Shape: (B, C, H, W)
            y_true (torch.Tensor): Target feature map. Shape: (B, C, H, W)
            
        Returns:
            torch.Tensor: Scalar loss value averaged over the batch.
        """
        B = y_true.size(0)
        x_flat = y_true.view(B, -1)
        y_flat = y_pred.view(B, -1)
        mx = torch.mean(x_flat, dim=1, keepdim=True)
        my = torch.mean(y_flat, dim=1, keepdim=True)
        xm = x_flat - mx
        ym = y_flat - my
        r_num = torch.sum(xm * ym, dim=1)
        r_den = torch.sqrt(torch.sum(xm ** 2, dim=1) * torch.sum(ym ** 2, dim=1) + self.eps)
        r = r_num / r_den
        r = torch.clamp(r, min=-1.0 + self.eps, max=1.0 - self.eps)
        loss = 1.0 - r
        return torch.mean(loss)