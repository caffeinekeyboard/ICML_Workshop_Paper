import torch
import pytest

from model.gumnet_spatial_alignment_mp import GumNetSpatialAlignmentMP

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def regressor():
    model = GumNetSpatialAlignmentMP(input_dim=100352)
    model.eval()
    return model

@pytest.fixture
def dummy_data():
    batch_size = 4
    single_corr_dim = 50176
    c_ab = torch.randn(batch_size, single_corr_dim)
    c_ba = torch.randn(batch_size, single_corr_dim)
    source_image = torch.randn(batch_size, 1, 192, 192)
    return c_ab, c_ba, source_image

# -----------------------------------------------------------------------------
# 1. INITIALIZATION & SETUP TESTS
# -----------------------------------------------------------------------------

def test_model_initialization(regressor):
    assert isinstance(regressor, GumNetSpatialAlignmentMP)
    assert hasattr(regressor, 'fc1')
    assert hasattr(regressor, 'fc2')
    assert hasattr(regressor, 'fc_out')

def test_identity_initialization_enforcement(regressor):
    assert torch.all(regressor.fc_out.weight == 0.0), \
        "fc_out weights must be strictly zero for identity initialization."
    assert torch.all(regressor.fc_out.bias == 0.0), \
        "fc_out biases must be strictly zero for identity initialization."

# -----------------------------------------------------------------------------
# 2. FUNCTIONAL TESTS (DIMENSIONALITY & CORRECTNESS)
# -----------------------------------------------------------------------------

def test_forward_pass_dimensions(regressor, dummy_data):
    c_ab, c_ba, source_image = dummy_data
    
    with torch.no_grad():
        warped_image, affine_matrix = regressor(c_ab, c_ba, source_image)

    assert warped_image.shape == source_image.shape, \
        f"Expected warped image shape {source_image.shape}, got {warped_image.shape}"
    
    expected_matrix_shape = (source_image.size(0), 2, 3)

    assert affine_matrix.shape == expected_matrix_shape, \
        f"Expected affine matrix shape {expected_matrix_shape}, got {affine_matrix.shape}"

def test_forward_pass_identity_warp(regressor, dummy_data):
    c_ab, c_ba, source_image = dummy_data
    
    with torch.no_grad():
        warped_image, affine_matrix = regressor(c_ab, c_ba, source_image)

    B = source_image.size(0)
    expected_identity = torch.tensor([[[1.0, 0.0, 0.0], 
                                       [0.0, 1.0, 0.0]]]).repeat(B, 1, 1).to(affine_matrix.device)
    
    assert torch.allclose(affine_matrix, expected_identity, atol=1e-6), \
        "Initial affine matrix did not mathematically resolve to the Identity matrix."
    assert torch.allclose(warped_image, source_image, atol=1e-3), \
        "Warped image does not match source image despite an Identity transformation matrix."

# -----------------------------------------------------------------------------
# 3. TRAINING & GRAPH INTEGRITY TESTS
# -----------------------------------------------------------------------------

def test_zero_init_gradient_bottleneck(dummy_data):
    model = GumNetSpatialAlignmentMP(input_dim=100352)
    model.train()
    c_ab, c_ba, source_image = dummy_data
    warped_image, _ = model(c_ab, c_ba, source_image)
    loss = warped_image.sum()
    loss.backward()

    assert model.fc_out.weight.grad is not None
    assert torch.sum(torch.abs(model.fc_out.weight.grad)) > 0, \
        "fc_out did not receive gradients."
    assert model.fc1.weight.grad is not None
    assert torch.sum(torch.abs(model.fc1.weight.grad)) == 0, \
        "fc1 unexpectedly received gradients despite W_out being zero. Math is broken."

def test_full_graph_differentiability(dummy_data):
    model = GumNetSpatialAlignmentMP(input_dim=100352)
    model.train()
    
    with torch.no_grad():
        model.fc_out.weight.fill_(0.01) 
    
    c_ab, c_ba, source_image = dummy_data
    source_image.requires_grad = True
    warped_image, _ = model(c_ab, c_ba, source_image)
    loss = warped_image.sum()
    loss.backward()

    assert model.fc1.weight.grad is not None
    grad_value = model.fc1.weight.grad
    assert torch.sum(torch.abs(grad_value)) > 0, \
        "Gradients failed to reach fc1 after breaking the zero-initialization bottleneck."
    assert source_image.grad is not None
    source_grad = source_image.grad
    assert torch.sum(torch.abs(source_grad)) > 0, \
        "Spatial transformer grid_sample did not backpropagate to the source image."

# -----------------------------------------------------------------------------
# 4. HARDWARE & ECOSYSTEM COMPATIBILITY TESTS
# -----------------------------------------------------------------------------

def test_device_agnosticism(dummy_data):
    c_ab, c_ba, source_image = dummy_data
    model = GumNetSpatialAlignmentMP(input_dim=100352)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = model.to(device)
        c_ab = c_ab.to(device)
        c_ba = c_ba.to(device)
        source_image = source_image.to(device)
        
        with torch.no_grad():
            warped, affine = model(c_ab, c_ba, source_image)

        assert warped.device.type == device.type
        assert affine.device.type == device.type

    except Exception as e:
        pytest.fail(f"Model failed to execute after being moved to device {device}. Error: {e}")

def test_variable_batch_sizes(regressor):
    single_corr_dim = 50176
    batch_1_cab = torch.randn(1, single_corr_dim)
    batch_1_cba = torch.randn(1, single_corr_dim)
    batch_1_img = torch.randn(1, 1, 192, 192)
    batch_8_cab = torch.randn(8, single_corr_dim)
    batch_8_cba = torch.randn(8, single_corr_dim)
    batch_8_img = torch.randn(8, 1, 192, 192)

    with torch.no_grad():
        out_w1, out_a1 = regressor(batch_1_cab, batch_1_cba, batch_1_img)
        out_w8, out_a8 = regressor(batch_8_cab, batch_8_cba, batch_8_img)

    assert out_w1.shape == (1, 1, 192, 192)
    assert out_a1.shape == (1, 2, 3)
    assert out_w8.shape == (8, 1, 192, 192)
    assert out_a8.shape == (8, 2, 3)

# -----------------------------------------------------------------------------
# 5. MATHEMATICAL BOUNDS TESTING
# -----------------------------------------------------------------------------

def test_transformation_bounds():
    model = GumNetSpatialAlignmentMP(input_dim=100352)

    def force_max_hook(module, input, output):
        return torch.full_like(output, 100.0)
    
    handle = model.fc_out.register_forward_hook(force_max_hook)
    c_ab = torch.randn(1, 50176)
    c_ba = torch.randn(1, 50176)
    source = torch.randn(1, 1, 192, 192)
    
    with torch.no_grad():
        _, affine_matrix = model(c_ab, c_ba, source)
        
    handle.remove()
    expected_max_affine = torch.tensor([[[-1.0,  0.0,  1.0], 
                                         [ 0.0, -1.0,  1.0]]])
    
    assert torch.allclose(affine_matrix, expected_max_affine, atol=1e-5), \
        "Affine matrix mathematical bounds failed when Sigmoid output is 1.0."