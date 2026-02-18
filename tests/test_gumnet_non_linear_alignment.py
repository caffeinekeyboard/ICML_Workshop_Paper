import torch
import pytest
import torch.nn as nn

from model.gumnet_non_linear_alignment import GumNetNonLinearAlignment

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def nonlinear_alignment():
    model = GumNetNonLinearAlignment(input_dim=8192, grid_size=4)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 1. INITIALIZATION & SETUP TESTS
# -----------------------------------------------------------------------------

def test_model_initialization(nonlinear_alignment):
    assert isinstance(nonlinear_alignment, GumNetNonLinearAlignment)
    assert hasattr(nonlinear_alignment, 'fc1')
    assert hasattr(nonlinear_alignment, 'fc2')
    assert hasattr(nonlinear_alignment, 'fc_out')
    assert hasattr(nonlinear_alignment, 'grid_size')

def test_identity_initialization(nonlinear_alignment):
    out_weight = nonlinear_alignment.fc_out.weight
    out_bias = nonlinear_alignment.fc_out.bias

    assert torch.all(out_weight == 0), \
        "Output layer weights are not zero-initialized for identity transformation."
    assert torch.all(out_bias == 0), \
        "Output layer biases are not zero-initialized for identity transformation."

# -----------------------------------------------------------------------------
# 2. FUNCTIONAL TESTS (DIMENSIONALITY & CORRECTNESS)
# -----------------------------------------------------------------------------

def test_forward_pass_dimensions(nonlinear_alignment):
    batch_size = 2
    feature_dim = 4096
    channels = 1
    spatial = 192
    grid_size = 4
    
    c_ab = torch.randn(batch_size, feature_dim)
    c_ba = torch.randn(batch_size, feature_dim)
    source_image = torch.randn(batch_size, channels, spatial, spatial)
    
    with torch.no_grad():
        warped_image, control_points = nonlinear_alignment(c_ab, c_ba, source_image)

    expected_warped_shape = (batch_size, channels, spatial, spatial)
    expected_cp_shape = (batch_size, 2, grid_size, grid_size)

    assert warped_image.shape == expected_warped_shape, \
        f"Expected warped_image shape {expected_warped_shape}, but got {warped_image.shape}"
    assert control_points.shape == expected_cp_shape, \
        f"Expected control_points shape {expected_cp_shape}, but got {control_points.shape}"

# -----------------------------------------------------------------------------
# 3. TRAINING & GRAPH INTEGRITY TESTS
# -----------------------------------------------------------------------------

def test_backward_pass_and_differentiability():
    model = GumNetNonLinearAlignment(input_dim=8192, grid_size=4)
    nn.init.normal_(model.fc_out.weight)
    
    model.train()
    
    c_ab = torch.randn(2, 4096, requires_grad=True)
    c_ba = torch.randn(2, 4096, requires_grad=True)
    source_image = torch.randn(2, 1, 192, 192, requires_grad=True)
    
    warped_image, control_points = model(c_ab, c_ba, source_image)
    dummy_loss = warped_image.sum() + control_points.sum()
    dummy_loss.backward()

    assert c_ab.grad is not None, "Gradients did not reach input c_ab."
    assert c_ba.grad is not None, "Gradients did not reach input c_ba."
    assert source_image.grad is not None, "Gradients did not reach input source_image."

    fc1_grad = model.fc1.weight.grad
    fc2_grad = model.fc2.weight.grad
    fc_out_grad = model.fc_out.weight.grad

    assert fc1_grad is not None and torch.sum(torch.abs(fc1_grad)) > 0, \
        "Gradients failed to update fc1."
    assert fc2_grad is not None and torch.sum(torch.abs(fc2_grad)) > 0, \
        "Gradients failed to update fc2."
    assert fc_out_grad is not None and torch.sum(torch.abs(fc_out_grad)) > 0, \
        "Gradients failed to update fc_out."

# -----------------------------------------------------------------------------
# 4. HARDWARE & ECOSYSTEM COMPATIBILITY TESTS
# -----------------------------------------------------------------------------

def test_device_agnosticism():
    model = GumNetNonLinearAlignment(input_dim=8192, grid_size=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = model.to(device)
        c_ab = torch.randn(2, 4096).to(device)
        c_ba = torch.randn(2, 4096).to(device)
        source_image = torch.randn(2, 1, 192, 192).to(device)
        
        with torch.no_grad():
            warped_image, control_points = model(c_ab, c_ba, source_image)
            
        assert warped_image.device.type == device.type, "Output warped_image did not remain on the target device."
        assert control_points.device.type == device.type, "Output control_points did not remain on the target device."
        
        assert warped_image.device == source_image.device, "Output warped_image jumped to a different device."
        assert control_points.device == source_image.device, "Output control_points jumped to a different device."
        
    except Exception as e:
        pytest.fail(f"Model failed to execute after being moved to device {device}. Error: {e}")

def test_variable_batch_sizes(nonlinear_alignment):
    batch_1_c_ab = torch.randn(1, 4096)
    batch_1_c_ba = torch.randn(1, 4096)
    batch_1_img = torch.randn(1, 1, 192, 192)
    
    batch_4_c_ab = torch.randn(4, 4096)
    batch_4_c_ba = torch.randn(4, 4096)
    batch_4_img = torch.randn(4, 1, 192, 192)

    with torch.no_grad():
        warp_1, cp_1 = nonlinear_alignment(batch_1_c_ab, batch_1_c_ba, batch_1_img)
        warp_4, cp_4 = nonlinear_alignment(batch_4_c_ab, batch_4_c_ba, batch_4_img)

    assert warp_1.shape == (1, 1, 192, 192), f"Failed warped_image on Batch Size 1: {warp_1.shape}"
    assert cp_1.shape == (1, 2, 4, 4), f"Failed control_points on Batch Size 1: {cp_1.shape}"
    
    assert warp_4.shape == (4, 1, 192, 192), f"Failed warped_image on Batch Size 4: {warp_4.shape}" 
    assert cp_4.shape == (4, 2, 4, 4), f"Failed control_points on Batch Size 4: {cp_4.shape}"