import pytest
import torch
import torch.testing as tt

from model.layers.feature_correlation_2d import FeatureCorrelation2D

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def correlation_layer():
    """Fixture to instantiate the layer for tests."""
    return FeatureCorrelation2D()

# -----------------------------------------------------------------------------
# 1. MATHEMATICAL CORRECTNESS & DIMENSIONALITY TESTS
# -----------------------------------------------------------------------------

def test_feature_correlation_output_shape(correlation_layer):
    B, C, H, W = 2, 512, 14, 14
    f_A = torch.randn(B, C, H, W)
    f_B = torch.randn(B, C, H, W)
    output = correlation_layer(f_A, f_B)
    expected_shape = (B, H * W, H, W)
    
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

def test_feature_correlation_mathematical_correctness(correlation_layer):
    B, C, H, W = 2, 3, 4, 4
    f_A = torch.randn(B, C, H, W)
    f_B = torch.randn(B, C, H, W)
    optimized_output = correlation_layer(f_A, f_B)
    naive_expected = torch.zeros(B, H * W, H, W)
    f_A_flat = f_A.view(B, C, -1)  # shape: [B, C, 16]
    
    for b in range(B):
        for i in range(H * W):
            for y_B in range(H):
                for x_B in range(W):
                    vec_A = f_A_flat[b, :, i]
                    vec_B = f_B[b, :, y_B, x_B]
                    naive_expected[b, i, y_B, x_B] = torch.dot(vec_A, vec_B)
                    
    tt.assert_close(optimized_output, naive_expected, rtol=1e-5, atol=1e-5)

# -----------------------------------------------------------------------------
# 2. AUTODIFF & GRADIENT TESTS
# -----------------------------------------------------------------------------

def test_feature_correlation_autograd(correlation_layer):
    B, C, H, W = 1, 16, 8, 8
    f_A = torch.randn(B, C, H, W, requires_grad=True)
    f_B = torch.randn(B, C, H, W, requires_grad=True)
    output = correlation_layer(f_A, f_B)
    loss = output.sum()
    loss.backward()
    
    assert f_A.grad is not None, "Gradient did not flow back to f_A"
    assert f_B.grad is not None, "Gradient did not flow back to f_B"
    assert f_A.grad.shape == f_A.shape
    assert f_B.grad.shape == f_B.shape
    assert not torch.allclose(f_A.grad, torch.zeros_like(f_A.grad))

# -----------------------------------------------------------------------------
# 3. MISMATCHED INPUTS TESTS
# -----------------------------------------------------------------------------

def test_feature_correlation_mismatched_inputs(correlation_layer):
    B, C, H, W = 2, 512, 14, 14
    f_A_standard = torch.randn(B, C, H, W)
    f_B_bad_batch = torch.randn(B + 1, C, H, W)

    with pytest.raises(AssertionError, match="Batch size and Channel dimensions must match"):
        correlation_layer(f_A_standard, f_B_bad_batch)

    f_B_bad_channels = torch.randn(B, C + 1, H, W)
    with pytest.raises(AssertionError, match="Batch size and Channel dimensions must match"):
        correlation_layer(f_A_standard, f_B_bad_channels)
        
# ==========================================
# 4. GPU / CUDA SAFETY TESTS
# ==========================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this system.")
def test_feature_correlation_cuda_forward(correlation_layer):
    device = torch.device('cuda')
    correlation_layer.to(device)
    B, C, H, W = 2, 512, 14, 14
    f_A = torch.randn(B, C, H, W, device=device)
    f_B = torch.randn(B, C, H, W, device=device)
    output = correlation_layer(f_A, f_B)
    
    assert output.is_cuda, "Output tensor is not on the CUDA device."
    
    expected_shape = (B, H * W, H, W)
    
    assert output.shape == expected_shape

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this system.")
def test_feature_correlation_cuda_backward(correlation_layer):
    device = torch.device('cuda')
    correlation_layer.to(device)
    B, C, H, W = 2, 64, 14, 14
    f_A = torch.randn(B, C, H, W, device=device, requires_grad=True)
    f_B = torch.randn(B, C, H, W, device=device, requires_grad=True)
    output = correlation_layer(f_A, f_B)
    loss = output.sum()
    loss.backward()
    
    assert f_A.grad is not None and f_A.grad.is_cuda
    assert f_B.grad is not None and f_B.grad.is_cuda

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this system.")
def test_feature_correlation_cuda_half_precision(correlation_layer):
    device = torch.device('cuda')
    correlation_layer.to(device)
    B, C, H, W = 2, 512, 14, 14
    f_A = torch.randn(B, C, H, W, device=device).half()
    f_B = torch.randn(B, C, H, W, device=device).half()
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = correlation_layer(f_A, f_B)

    assert output.dtype == torch.float16, f"Expected float16, got {output.dtype}"
    assert not torch.isnan(output).any(), "NaNs detected during FP16 computation."