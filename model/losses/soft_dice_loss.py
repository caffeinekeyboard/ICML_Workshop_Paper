import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss (F1 Score Loss).
    
    The Soft Dice Loss is a smooth, differentiable version of the Dice coefficient, commonly used for semantic segmentation and alignment tasks. 
    It measures the overlap between predicted and ground truth regions, with values ranging from 0 (no overlap) to 1 (perfect overlap).
    The loss is computed as 1 - dice_score.
    
    This loss is particularly effective for handling class imbalance and encouraging the model to optimize for intersection-over-union metrics.
    
    Args:
        eps (float, optional): Small epsilon value for numerical stability to prevent
            division by zero. Defaults to 1e-8.
    """
    def __init__(self, eps=1e-8):
        super(SoftDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Computes the Soft Dice Loss.
        
        The Dice coefficient is calculated as:
            Dice = (2 * intersection) / (cardinality + eps)
        where intersection is the element-wise product sum and cardinality is the sum of all predicted and true values.
        
        Args:
            y_pred (torch.Tensor): Predicted feature map or logits. Shape: (B, C, H, W) or (B, ...)
            y_true (torch.Tensor): Ground truth feature map or labels. Shape: (B, C, H, W) or (B, ...)
            
        Returns:
            torch.Tensor: Scalar loss value (1 - average Dice score) averaged over the batch.
        """
        B = y_true.size(0)
        p_flat = y_pred.view(B, -1)
        t_flat = y_true.view(B, -1)
        intersection = torch.sum(p_flat * t_flat, dim=1)
        cardinality = torch.sum(p_flat, dim=1) + torch.sum(t_flat, dim=1)
        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        
        return torch.mean(1.0 - dice_score)