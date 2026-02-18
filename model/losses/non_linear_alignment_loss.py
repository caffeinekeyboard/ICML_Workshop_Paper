import torch
import torch.nn as nn

class NonLinearAlignmentLoss(nn.Module):
    """
    Combined Loss for Non-Linear Alignment Training.
    
    The total loss is computed as:
        L_total = L_dice + lambda_reg * L_smoothness
    where L_dice measures alignment quality and L_smoothness enforces grid regularity.
    
    Args:
        lambda_reg (float, optional): Weight for the smoothness regularization term.
            Controls the balance between alignment quality and deformation smoothness.
            Defaults to 0.1.
        eps (float, optional): Small epsilon value for numerical stability to prevent division by zero.
            Defaults to 1e-8.
    """
    def __init__(self, lambda_reg=0.1, eps=1e-8):
        """
        Initializes the NonLinearAlignmentLoss module.
        
        Args:
            lambda_reg (float, optional): Regularization weight. Defaults to 0.1.
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-8.
        """
        super(NonLinearAlignmentLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.eps = eps

    def compute_image_dice(self, y_pred, y_true):
        """
        Computes the Soft Dice Loss for alignment quality.
        
        Args:
            y_pred (torch.Tensor): Warped source image. Shape: (B, C, H, W)
            y_true (torch.Tensor): Target image. Shape: (B, C, H, W)
            
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

    def compute_grid_smoothness(self, control_points):
        """
        Computes the smoothness regularization loss for the deformation grid.
        
        Args:
            control_points (torch.Tensor): Displacement vectors at control points.
                Shape: (B, 2, grid_size, grid_size)
            
        Returns:
            torch.Tensor: Scalar smoothness loss measuring gradient magnitude.
        """
        dx = control_points[:, :, :, 1:] - control_points[:, :, :, :-1]
        dy = control_points[:, :, 1:, :] - control_points[:, :, :-1, :]
        smoothness_loss = torch.mean(dx ** 2) + torch.mean(dy ** 2)
        
        return smoothness_loss

    def forward(self, warped_image, target_image, control_points):
        """
        Computes the total non-linear alignment loss.
        
        Args:
            warped_image (torch.Tensor): Warped source image after deformation. Shape: (B, C, H, W)
            target_image (torch.Tensor): Target template image. Shape: (B, C, H, W)
            control_points (torch.Tensor): Predicted control point displacements. Shape: (B, 2, grid_size, grid_size)
            
        Returns:
            tuple: A tuple of three scalar tensors:
                - total_loss (torch.Tensor): Weighted combination of dice and regularization losses.
                - dice_loss (torch.Tensor): Image alignment loss (1 - Dice score).
                - reg_loss (torch.Tensor): Grid smoothness regularization loss.
        """
        dice_loss = self.compute_image_dice(warped_image, target_image)
        reg_loss = self.compute_grid_smoothness(control_points)
        total_loss = dice_loss + (self.lambda_reg * reg_loss)
        
        return total_loss, dice_loss, reg_loss