import pytest
import torch
import torch.testing as tt

from model.losses.soft_dice_loss import SoftDiceLoss

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def soft_dice_loss():
    return SoftDiceLoss(eps=1e-8)

# -----------------------------------------------------------------------------
# 1. MATHEMATICAL CORRECTNESS & DIMENSIONALITY TESTS
# -----------------------------------------------------------------------------

def test_soft_dice_loss_output_shape(soft_dice_loss):
    B, C, H, W = 4, 1, 64, 64
    y_pred = torch.rand(B, C, H, W)
    y_true = torch.rand(B, C, H, W)
    loss = soft_dice_loss(y_pred, y_true)
    
    assert loss.dim() == 0, f"Expected a scalar (0 dims), got {loss.dim()} dims"

def test_soft_dice_loss_identical_inputs(soft_dice_loss):
    B, C, H, W = 2, 32, 14, 14
    y_true = torch.ones(B, C, H, W)
    loss = soft_dice_loss(y_true, y_true)
    
    tt.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

def test_soft_dice_loss_orthogonal_inputs(soft_dice_loss):
    B, C, H, W = 2, 1, 16, 16
    y_pred = torch.ones(B, C, H, W)
    y_true = torch.zeros(B, C, H, W)
    
    loss = soft_dice_loss(y_pred, y_true)
    
    tt.assert_close(loss, torch.tensor(1.0), atol=1e-5, rtol=1e-5)

def test_soft_dice_loss_manual_calc(soft_dice_loss):
    B, C, H, W = 2, 1, 4, 4
    y_pred = torch.rand(B, C, H, W)
    y_true = torch.rand(B, C, H, W)
    loss_tensor = soft_dice_loss(y_pred, y_true)
    expected_losses = []
    
    for b in range(B):
        pred_flat = y_pred[b].flatten()
        true_flat = y_true[b].flatten()
        intersection = torch.sum(pred_flat * true_flat)
        cardinality = torch.sum(pred_flat) + torch.sum(true_flat)
        dice_score = (2.0 * intersection + 1e-8) / (cardinality + 1e-8)
        expected_losses.append(1.0 - dice_score)
        
    expected_loss_avg = torch.stack(expected_losses).mean()
    
    tt.assert_close(loss_tensor, expected_loss_avg, atol=1e-6, rtol=1e-6)

# -----------------------------------------------------------------------------
# 2. AUTODIFF & GRADIENT TESTS
# -----------------------------------------------------------------------------

def test_soft_dice_loss_autograd(soft_dice_loss):
    B, C, H, W = 2, 1, 14, 14
    y_pred = torch.rand(B, C, H, W, requires_grad=True)
    y_true = torch.rand(B, C, H, W, requires_grad=True)
    loss = soft_dice_loss(y_pred, y_true)
    loss.backward()
    
    assert y_pred.grad is not None, "Gradient did not flow back to y_pred"
    assert y_true.grad is not None, "Gradient did not flow back to y_true"
    assert y_pred.grad.shape == y_pred.shape
    assert y_true.grad.shape == y_true.shape
    assert not torch.allclose(y_pred.grad, torch.zeros_like(y_pred.grad))

# -----------------------------------------------------------------------------
# 3. NUMERICAL STABILITY & EDGE CASES TESTS
# -----------------------------------------------------------------------------

def test_soft_dice_loss_zero_values(soft_dice_loss):
    B, C, H, W = 2, 1, 14, 14
    y_pred = torch.zeros(B, C, H, W)
    y_true = torch.zeros(B, C, H, W)
    loss = soft_dice_loss(y_pred, y_true)

    assert not torch.isnan(loss), "Loss computed as NaN due to zero values."
    assert not torch.isinf(loss), "Loss computed as Inf due to zero values."
    tt.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

def test_soft_dice_loss_mismatched_inputs(soft_dice_loss):
    B, C, H, W = 2, 1, 14, 14
    y_pred = torch.rand(B, C, H, W)
    y_true = torch.rand(B, C, H, W + 1)
    
    with pytest.raises(RuntimeError):
        soft_dice_loss(y_pred, y_true)

def test_soft_dice_loss_very_small_values(soft_dice_loss):
    B, C, H, W = 2, 1, 14, 14
    y_pred = torch.rand(B, C, H, W) * 1e-10
    y_true = torch.rand(B, C, H, W) * 1e-10
    loss = soft_dice_loss(y_pred, y_true)
    
    assert not torch.isnan(loss), "Loss is NaN with very small values"
    assert not torch.isinf(loss), "Loss is Inf with very small values"
    assert loss >= 0.0, "Loss should be non-negative"

# -----------------------------------------------------------------------------
# 4. GPU / CUDA & PRECISION TESTS
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this system.")
def test_soft_dice_loss_cuda_forward(soft_dice_loss):
    device = torch.device('cuda')
    soft_dice_loss.to(device)
    B, C, H, W = 4, 128, 14, 14
    y_pred = torch.rand(B, C, H, W, device=device)
    y_true = torch.rand(B, C, H, W, device=device)
    loss = soft_dice_loss(y_pred, y_true)
    
    assert loss.is_cuda, "Output tensor is not on the CUDA device."
    assert loss.dim() == 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this system.")
def test_soft_dice_loss_cuda_backward(soft_dice_loss):
    device = torch.device('cuda')
    soft_dice_loss.to(device)
    B, C, H, W = 4, 128, 14, 14
    y_pred = torch.rand(B, C, H, W, device=device, requires_grad=True)
    y_true = torch.rand(B, C, H, W, device=device, requires_grad=True)
    loss = soft_dice_loss(y_pred, y_true)
    loss.backward()
    
    assert y_pred.grad is not None and y_pred.grad.is_cuda
    assert y_true.grad is not None and y_true.grad.is_cuda

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this system.")
def test_soft_dice_loss_cuda_half_precision(soft_dice_loss):
    device = torch.device('cuda')
    soft_dice_loss.to(device)
    B, C, H, W = 2, 512, 14, 14
    y_pred = torch.rand(B, C, H, W, device=device).half()
    y_true = torch.rand(B, C, H, W, device=device).half()
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss = soft_dice_loss(y_pred, y_true)

    assert not torch.isnan(loss).any(), "NaNs detected during Mixed Precision FP16 computation."
    assert not torch.isinf(loss).any(), "Infs detected during Mixed Precision FP16 computation."
