import pytest
import torch
import torch.testing as tt

from model.losses.pearson_correlation_loss import PearsonCorrelationLoss

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def pearson_loss():
    return PearsonCorrelationLoss(eps=1e-8)

# -----------------------------------------------------------------------------
# 1. MATHEMATICAL CORRECTNESS & DIMENSIONALITY TESTS
# -----------------------------------------------------------------------------

def test_pearson_loss_output_shape(pearson_loss):
    B, C, H, W = 4, 1, 64, 64
    y_pred = torch.randn(B, C, H, W)
    y_true = torch.randn(B, C, H, W)
    loss = pearson_loss(y_pred, y_true)
    
    assert loss.dim() == 0, f"Expected a scalar (0 dims), got {loss.dim()} dims"

def test_pearson_loss_identical_inputs(pearson_loss):
    B, C, H, W = 2, 32, 14, 14
    y_true = torch.randn(B, C, H, W)
    loss = pearson_loss(y_true, y_true)
    
    tt.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

def test_pearson_loss_opposite_inputs(pearson_loss):
    """If inputs are perfectly anti-correlated (opposite), r=-1, loss should be 2."""
    B, C, H, W = 2, 32, 14, 14
    y_true = torch.randn(B, C, H, W)
    y_pred = -y_true
    loss = pearson_loss(y_pred, y_true)
    
    tt.assert_close(loss, torch.tensor(2.0), atol=1e-5, rtol=1e-5)

def test_pearson_loss_manual_calc(pearson_loss):
    B, C, H, W = 2, 1, 4, 4
    y_pred = torch.randn(B, C, H, W)
    y_true = torch.randn(B, C, H, W)
    loss_tensor = pearson_loss(y_pred, y_true)
    expected_losses = []
    
    for b in range(B):
        pred_flat = y_pred[b].flatten()
        true_flat = y_true[b].flatten()
        pred_mean = pred_flat.mean()
        true_mean = true_flat.mean()
        pred_ctr = pred_flat - pred_mean
        true_ctr = true_flat - true_mean
        cov = torch.sum(pred_ctr * true_ctr)
        std = torch.sqrt(torch.sum(pred_ctr**2) * torch.sum(true_ctr**2) + 1e-8)
        r = torch.clamp(cov / std, min=-1.0+1e-8, max=1.0-1e-8)
        expected_losses.append(1.0 - r)
        
    expected_loss_avg = torch.stack(expected_losses).mean()
    
    tt.assert_close(loss_tensor, expected_loss_avg, atol=1e-6, rtol=1e-6)

# -----------------------------------------------------------------------------
# 2. AUTODIFF & GRADIENT TESTS
# -----------------------------------------------------------------------------

def test_pearson_loss_autograd(pearson_loss):
    """Ensure gradients flow properly backward through both inputs."""
    B, C, H, W = 2, 1, 14, 14
    y_pred = torch.randn(B, C, H, W, requires_grad=True)
    y_true = torch.randn(B, C, H, W, requires_grad=True)
    loss = pearson_loss(y_pred, y_true)
    loss.backward()
    
    assert y_pred.grad is not None, "Gradient did not flow back to y_pred"
    assert y_true.grad is not None, "Gradient did not flow back to y_true"
    assert y_pred.grad.shape == y_pred.shape
    assert y_true.grad.shape == y_true.shape
    assert not torch.allclose(y_pred.grad, torch.zeros_like(y_pred.grad))

# -----------------------------------------------------------------------------
# 3. NUMERICAL STABILITY & EDGE CASES TESTS
# -----------------------------------------------------------------------------

def test_pearson_loss_zero_variance(pearson_loss):
    """Test behavior when inputs have 0 variance (constant flat maps). EPS prevents NaNs."""
    B, C, H, W = 2, 1, 14, 14
    y_pred = torch.ones(B, C, H, W) * 5.0
    y_true = torch.ones(B, C, H, W) * 2.0
    loss = pearson_loss(y_pred, y_true)

    assert not torch.isnan(loss), "Loss computed as NaN due to zero variance."
    assert not torch.isinf(loss), "Loss computed as Inf due to zero variance."
    tt.assert_close(loss, torch.tensor(1.0), atol=1e-6, rtol=1e-6)

def test_pearson_loss_mismatched_inputs(pearson_loss):
    B, C, H, W = 2, 1, 14, 14
    y_pred = torch.randn(B, C, H, W)
    y_true = torch.randn(B, C, H, W + 1)
    
    with pytest.raises(RuntimeError):
        pearson_loss(y_pred, y_true)

# -----------------------------------------------------------------------------
# 4. GPU / CUDA & PRECISION TESTS
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this system.")
def test_pearson_loss_cuda_forward(pearson_loss):
    device = torch.device('cuda')
    pearson_loss.to(device)
    B, C, H, W = 4, 128, 14, 14
    y_pred = torch.randn(B, C, H, W, device=device)
    y_true = torch.randn(B, C, H, W, device=device)
    loss = pearson_loss(y_pred, y_true)
    
    assert loss.is_cuda, "Output tensor is not on the CUDA device."
    assert loss.dim() == 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this system.")
def test_pearson_loss_cuda_backward(pearson_loss):
    device = torch.device('cuda')
    pearson_loss.to(device)
    B, C, H, W = 4, 128, 14, 14
    y_pred = torch.randn(B, C, H, W, device=device, requires_grad=True)
    y_true = torch.randn(B, C, H, W, device=device, requires_grad=True)
    loss = pearson_loss(y_pred, y_true)
    loss.backward()
    
    assert y_pred.grad is not None and y_pred.grad.is_cuda
    assert y_true.grad is not None and y_true.grad.is_cuda

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this system.")
def test_pearson_loss_cuda_half_precision(pearson_loss):
    device = torch.device('cuda')
    pearson_loss.to(device)
    B, C, H, W = 2, 512, 14, 14
    y_pred = torch.randn(B, C, H, W, device=device).half()
    y_true = torch.randn(B, C, H, W, device=device).half()
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss = pearson_loss(y_pred, y_true)

    assert not torch.isnan(loss).any(), "NaNs detected during Mixed Precision FP16 computation."
    assert not torch.isinf(loss).any(), "Infs detected during Mixed Precision FP16 computation."