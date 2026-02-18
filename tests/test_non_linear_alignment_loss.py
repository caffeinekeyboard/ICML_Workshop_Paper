import pytest
import torch
import torch.testing as tt

from model.losses.non_linear_alignment_loss import NonLinearAlignmentLoss

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def non_linear_alignment_loss():
    return NonLinearAlignmentLoss(lambda_reg=0.1, eps=1e-8)

# -----------------------------------------------------------------------------
# 1. MATHEMATICAL CORRECTNESS & DIMENSIONALITY TESTS
# -----------------------------------------------------------------------------

def test_non_linear_alignment_loss_output_shape(non_linear_alignment_loss):
    """Test that the loss returns three scalar outputs."""
    B, C, H, W = 4, 1, 64, 64
    warped_image = torch.rand(B, C, H, W)
    target_image = torch.rand(B, C, H, W)
    control_points = torch.randn(B, 2, 8, 8)
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    
    assert total_loss.dim() == 0, f"Expected scalar (0 dims), got {total_loss.dim()} dims"
    assert dice_loss.dim() == 0, f"Expected scalar (0 dims), got {dice_loss.dim()} dims"
    assert reg_loss.dim() == 0, f"Expected scalar (0 dims), got {reg_loss.dim()} dims"

def test_non_linear_alignment_loss_identical_images(non_linear_alignment_loss):
    """When warped and target images are identical, dice loss should be 0."""
    B, C, H, W = 2, 32, 14, 14
    warped_image = torch.ones(B, C, H, W)
    target_image = torch.ones(B, C, H, W)
    control_points = torch.randn(B, 2, 8, 8)
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    
    tt.assert_close(dice_loss, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

def test_non_linear_alignment_loss_zero_control_points(non_linear_alignment_loss):
    """When control points are all zeros, smoothness loss should be 0."""
    B, C, H, W = 2, 1, 16, 16
    warped_image = torch.rand(B, C, H, W)
    target_image = torch.rand(B, C, H, W)
    control_points = torch.zeros(B, 2, 8, 8)
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    
    tt.assert_close(reg_loss, torch.tensor(0.0), atol=1e-8, rtol=1e-8)

def test_non_linear_alignment_loss_composition(non_linear_alignment_loss):
    """Test that total_loss = dice_loss + lambda_reg * reg_loss."""
    B, C, H, W = 2, 1, 16, 16
    warped_image = torch.rand(B, C, H, W)
    target_image = torch.rand(B, C, H, W)
    control_points = torch.randn(B, 2, 8, 8)
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    
    expected_total = dice_loss + non_linear_alignment_loss.lambda_reg * reg_loss
    tt.assert_close(total_loss, expected_total, atol=1e-6, rtol=1e-6)

def test_non_linear_alignment_loss_manual_dice_calc(non_linear_alignment_loss):
    """Manually verify dice loss computation."""
    B, C, H, W = 2, 1, 4, 4
    warped_image = torch.rand(B, C, H, W)
    target_image = torch.rand(B, C, H, W)
    control_points = torch.randn(B, 2, 4, 4)
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    expected_losses = []
    
    for b in range(B):
        pred_flat = warped_image[b].flatten()
        true_flat = target_image[b].flatten()
        intersection = torch.sum(pred_flat * true_flat)
        cardinality = torch.sum(pred_flat) + torch.sum(true_flat)
        dice_score = (2.0 * intersection + 1e-8) / (cardinality + 1e-8)
        expected_losses.append(1.0 - dice_score)
    
    expected_dice_loss = torch.stack(expected_losses).mean()
    tt.assert_close(dice_loss, expected_dice_loss, atol=1e-6, rtol=1e-6)

def test_non_linear_alignment_loss_manual_smoothness_calc(non_linear_alignment_loss):
    """Manually verify smoothness regularization computation."""
    B, C, H, W = 2, 1, 4, 4
    warped_image = torch.rand(B, C, H, W)
    target_image = torch.rand(B, C, H, W)
    control_points = torch.randn(B, 2, 4, 4)
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    
    # Manually compute smoothness
    dx = control_points[:, :, :, 1:] - control_points[:, :, :, :-1]
    dy = control_points[:, :, 1:, :] - control_points[:, :, :-1, :]
    expected_smoothness = torch.mean(dx ** 2) + torch.mean(dy ** 2)
    
    tt.assert_close(reg_loss, expected_smoothness, atol=1e-6, rtol=1e-6)

# -----------------------------------------------------------------------------
# 2. AUTODIFF & GRADIENT TESTS
# -----------------------------------------------------------------------------

def test_non_linear_alignment_loss_autograd_images(non_linear_alignment_loss):
    """Ensure gradients flow properly backward through images."""
    B, C, H, W = 2, 1, 14, 14
    warped_image = torch.rand(B, C, H, W, requires_grad=True)
    target_image = torch.rand(B, C, H, W, requires_grad=True)
    control_points = torch.randn(B, 2, 8, 8)
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    total_loss.backward()
    
    assert warped_image.grad is not None, "Gradient did not flow back to warped_image"
    assert target_image.grad is not None, "Gradient did not flow back to target_image"
    assert warped_image.grad.shape == warped_image.shape
    assert target_image.grad.shape == target_image.shape

def test_non_linear_alignment_loss_autograd_control_points(non_linear_alignment_loss):
    """Ensure gradients flow properly backward through control points."""
    B, C, H, W = 2, 1, 14, 14
    warped_image = torch.rand(B, C, H, W)
    target_image = torch.rand(B, C, H, W)
    control_points = torch.randn(B, 2, 8, 8, requires_grad=True)
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    total_loss.backward()
    
    assert control_points.grad is not None, "Gradient did not flow back to control_points"
    assert control_points.grad.shape == control_points.shape
    assert not torch.allclose(control_points.grad, torch.zeros_like(control_points.grad))

def test_non_linear_alignment_loss_autograd_all_inputs(non_linear_alignment_loss):
    """Ensure gradients flow properly for all inputs simultaneously."""
    B, C, H, W = 2, 1, 14, 14
    warped_image = torch.rand(B, C, H, W, requires_grad=True)
    target_image = torch.rand(B, C, H, W, requires_grad=True)
    control_points = torch.randn(B, 2, 8, 8, requires_grad=True)
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    total_loss.backward()
    
    assert warped_image.grad is not None and warped_image.grad.shape == warped_image.shape
    assert target_image.grad is not None and target_image.grad.shape == target_image.shape
    assert control_points.grad is not None and control_points.grad.shape == control_points.shape

# -----------------------------------------------------------------------------
# 3. NUMERICAL STABILITY & EDGE CASES TESTS
# -----------------------------------------------------------------------------

def test_non_linear_alignment_loss_zero_images(non_linear_alignment_loss):
    """Test behavior when both images are zero."""
    B, C, H, W = 2, 1, 14, 14
    warped_image = torch.zeros(B, C, H, W)
    target_image = torch.zeros(B, C, H, W)
    control_points = torch.randn(B, 2, 8, 8)
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    
    assert not torch.isnan(dice_loss), "Dice loss is NaN with zero images"
    assert not torch.isinf(dice_loss), "Dice loss is Inf with zero images"
    tt.assert_close(dice_loss, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

def test_non_linear_alignment_loss_mismatched_image_shapes(non_linear_alignment_loss):
    """Test that mismatched image shapes raise an error."""
    B, C, H, W = 2, 1, 14, 14
    warped_image = torch.rand(B, C, H, W)
    target_image = torch.rand(B, C, H, W + 1)
    control_points = torch.randn(B, 2, 8, 8)
    
    with pytest.raises(RuntimeError):
        non_linear_alignment_loss(warped_image, target_image, control_points)

def test_non_linear_alignment_loss_very_small_values(non_linear_alignment_loss):
    """Test stability with very small floating point values."""
    B, C, H, W = 2, 1, 14, 14
    warped_image = torch.rand(B, C, H, W) * 1e-10
    target_image = torch.rand(B, C, H, W) * 1e-10
    control_points = torch.randn(B, 2, 8, 8) * 1e-10
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    
    assert not torch.isnan(total_loss), "Total loss is NaN with very small values"
    assert not torch.isinf(total_loss), "Total loss is Inf with very small values"
    assert total_loss >= 0.0, "Total loss should be non-negative"

def test_non_linear_alignment_loss_large_control_point_deformations(non_linear_alignment_loss):
    """Test stability with large control point displacements."""
    B, C, H, W = 2, 1, 14, 14
    warped_image = torch.rand(B, C, H, W)
    target_image = torch.rand(B, C, H, W)
    control_points = torch.randn(B, 2, 8, 8) * 100.0  # Large displacements
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    
    assert not torch.isnan(total_loss), "Loss is NaN with large control point values"
    assert not torch.isinf(total_loss), "Loss is Inf with large control point values"

def test_non_linear_alignment_loss_lambda_reg_effect(non_linear_alignment_loss):
    """Test that lambda_reg properly scales the regularization term."""
    B, C, H, W = 2, 1, 14, 14
    warped_image = torch.rand(B, C, H, W)
    target_image = torch.rand(B, C, H, W)
    control_points = torch.randn(B, 2, 8, 8)
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    
    # Create a loss with different lambda_reg
    loss_high_lambda = NonLinearAlignmentLoss(lambda_reg=1.0, eps=1e-8)
    total_loss_high, dice_loss_high, reg_loss_high = loss_high_lambda(warped_image, target_image, control_points)
    
    # Dice loss should be the same
    tt.assert_close(dice_loss, dice_loss_high, atol=1e-6, rtol=1e-6)
    
    # Total loss should be different (higher with larger lambda_reg)
    assert total_loss < total_loss_high, "Higher lambda_reg should increase total loss"

# -----------------------------------------------------------------------------
# 4. GPU / CUDA & PRECISION TESTS
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this system.")
def test_non_linear_alignment_loss_cuda_forward(non_linear_alignment_loss):
    device = torch.device('cuda')
    non_linear_alignment_loss.to(device)
    B, C, H, W = 4, 128, 14, 14
    warped_image = torch.rand(B, C, H, W, device=device)
    target_image = torch.rand(B, C, H, W, device=device)
    control_points = torch.randn(B, 2, 8, 8, device=device)
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    
    assert total_loss.is_cuda, "Total loss is not on CUDA device"
    assert dice_loss.is_cuda, "Dice loss is not on CUDA device"
    assert reg_loss.is_cuda, "Regularization loss is not on CUDA device"
    assert total_loss.dim() == 0
    assert dice_loss.dim() == 0
    assert reg_loss.dim() == 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this system.")
def test_non_linear_alignment_loss_cuda_backward(non_linear_alignment_loss):
    device = torch.device('cuda')
    non_linear_alignment_loss.to(device)
    B, C, H, W = 4, 128, 14, 14
    warped_image = torch.rand(B, C, H, W, device=device, requires_grad=True)
    target_image = torch.rand(B, C, H, W, device=device, requires_grad=True)
    control_points = torch.randn(B, 2, 8, 8, device=device, requires_grad=True)
    
    total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    total_loss.backward()
    
    assert warped_image.grad is not None and warped_image.grad.is_cuda
    assert target_image.grad is not None and target_image.grad.is_cuda
    assert control_points.grad is not None and control_points.grad.is_cuda

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this system.")
def test_non_linear_alignment_loss_cuda_half_precision(non_linear_alignment_loss):
    device = torch.device('cuda')
    non_linear_alignment_loss.to(device)
    B, C, H, W = 2, 512, 14, 14
    warped_image = torch.rand(B, C, H, W, device=device).half()
    target_image = torch.rand(B, C, H, W, device=device).half()
    control_points = torch.randn(B, 2, 8, 8, device=device).half()
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        total_loss, dice_loss, reg_loss = non_linear_alignment_loss(warped_image, target_image, control_points)
    
    assert not torch.isnan(total_loss).any(), "NaNs detected in total loss during FP16 computation"
    assert not torch.isinf(total_loss).any(), "Infs detected in total loss during FP16 computation"
    assert not torch.isnan(dice_loss).any(), "NaNs detected in dice loss during FP16 computation"
    assert not torch.isinf(dice_loss).any(), "Infs detected in dice loss during FP16 computation"
