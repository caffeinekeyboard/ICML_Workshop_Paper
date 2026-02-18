import torch
import pytest

from model.gumnet_ap import GumNetAP

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def gumnet_model():
    model = GumNetAP(in_channels=1)
    model.eval()
    return model

@pytest.fixture
def gumnet_model_rgb():
    model = GumNetAP(in_channels=3)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 1. INITIALIZATION & SETUP TESTS
# -----------------------------------------------------------------------------

def test_model_initialization_default(gumnet_model):
    assert isinstance(gumnet_model, GumNetAP)
    assert hasattr(gumnet_model, 'feature_extractor')
    assert hasattr(gumnet_model, 'siamese_matcher')
    assert hasattr(gumnet_model, 'spatial_aligner')
    assert gumnet_model.feature_extractor is not None
    assert gumnet_model.siamese_matcher is not None
    assert gumnet_model.spatial_aligner is not None

# -----------------------------------------------------------------------------
# 2. FUNCTIONAL TESTS (DIMENSIONALITY & CORRECTNESS)
# -----------------------------------------------------------------------------

def test_forward_pass_dimensions_default(gumnet_model):
    batch_size = 2
    height, width = 192, 192
    channels = 1
    template = torch.randn(batch_size, channels, height, width)
    impression = torch.randn(batch_size, channels, height, width)

    with torch.no_grad():
        warped_impression, affine_matrix = gumnet_model(template, impression)

    expected_warped_shape = (batch_size, channels, height, width)
    expected_matrix_shape = (batch_size, 2, 3)

    assert warped_impression.shape == expected_warped_shape, \
        f"Expected warped_impression shape {expected_warped_shape}, got {warped_impression.shape}"
    assert affine_matrix.shape == expected_matrix_shape, \
        f"Expected affine_matrix shape {expected_matrix_shape}, got {affine_matrix.shape}"
    assert isinstance(warped_impression, torch.Tensor)
    assert isinstance(affine_matrix, torch.Tensor)

def test_forward_pass_dimensions_rgb(gumnet_model_rgb):
    batch_size = 2
    height, width = 192, 192
    channels = 3
    template = torch.randn(batch_size, channels, height, width)
    impression = torch.randn(batch_size, channels, height, width)

    with torch.no_grad():
        warped_impression, affine_matrix = gumnet_model_rgb(template, impression)

    expected_warped_shape = (batch_size, channels, height, width)
    expected_matrix_shape = (batch_size, 2, 3)

    assert warped_impression.shape == expected_warped_shape
    assert affine_matrix.shape == expected_matrix_shape

def test_forward_pass_single_batch(gumnet_model):
    batch_size = 1
    height, width = 192, 192
    channels = 1
    template = torch.randn(batch_size, channels, height, width)
    impression = torch.randn(batch_size, channels, height, width)

    with torch.no_grad():
        warped_impression, affine_matrix = gumnet_model(template, impression)

    expected_warped_shape = (batch_size, channels, height, width)
    expected_matrix_shape = (batch_size, 2, 3)

    assert warped_impression.shape == expected_warped_shape
    assert affine_matrix.shape == expected_matrix_shape

def test_forward_pass_large_batch(gumnet_model):
    batch_size = 8
    height, width = 192, 192
    channels = 1
    template = torch.randn(batch_size, channels, height, width)
    impression = torch.randn(batch_size, channels, height, width)

    with torch.no_grad():
        warped_impression, affine_matrix = gumnet_model(template, impression)

    expected_warped_shape = (batch_size, channels, height, width)
    expected_matrix_shape = (batch_size, 2, 3)

    assert warped_impression.shape == expected_warped_shape
    assert affine_matrix.shape == expected_matrix_shape
    
def test_affine_matrix_properties(gumnet_model):
    import torch.nn as nn
    nn.init.normal_(gumnet_model.spatial_aligner.fc_out.weight, std=0.01)
    batch_size = 4
    template = torch.randn(batch_size, 1, 192, 192)
    impression = torch.randn(batch_size, 1, 192, 192)

    with torch.no_grad():
        _, affine_matrix = gumnet_model(template, impression)

    assert torch.isfinite(affine_matrix).all(), "Affine matrix contains non-finite values"
    assert not torch.isnan(affine_matrix).any(), "Affine matrix contains NaN values"
    assert affine_matrix.shape == (batch_size, 2, 3), f"Expected shape {(batch_size, 2, 3)}, got {affine_matrix.shape}"

# -----------------------------------------------------------------------------
# 3. TRAINING & GRAPH INTEGRITY TESTS
# -----------------------------------------------------------------------------

def test_backward_pass_and_differentiability():
    import torch.nn as nn
    model = GumNetAP(in_channels=1)
    nn.init.normal_(model.spatial_aligner.fc_out.weight, std=0.01)
    model.train()
    template = torch.randn(2, 1, 192, 192, requires_grad=True)
    impression = torch.randn(2, 1, 192, 192, requires_grad=True)
    warped_impression, affine_matrix = model(template, impression)
    loss = warped_impression.sum() + affine_matrix.sum()
    loss.backward()

    assert template.grad is not None, "Gradients did not reach template input"
    assert impression.grad is not None, "Gradients did not reach impression input"
    assert torch.sum(torch.abs(template.grad)) > 0, "Template gradients are zero"
    assert torch.sum(torch.abs(impression.grad)) > 0, "Impression gradients are zero"

def test_model_train_eval_modes():
    model = GumNetAP(in_channels=1)
    template = torch.randn(2, 1, 192, 192)
    impression = torch.randn(2, 1, 192, 192)
    model.train()
    
    with torch.no_grad():
        train_warped, train_matrix = model(template, impression)

    model.eval()
    
    with torch.no_grad():
        eval_warped, eval_matrix = model(template, impression)

    assert torch.allclose(train_warped, eval_warped, atol=1e-6), \
        "Train/eval modes produce different warped impressions"
    assert torch.allclose(train_matrix, eval_matrix, atol=1e-6), \
        "Train/eval modes produce different affine matrices"

# -----------------------------------------------------------------------------
# 4. HARDWARE & ECOSYSTEM COMPATIBILITY TESTS
# -----------------------------------------------------------------------------

def test_device_agnosticism():
    model = GumNetAP(in_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = model.to(device)
        template = torch.randn(2, 1, 192, 192).to(device)
        impression = torch.randn(2, 1, 192, 192).to(device)

        with torch.no_grad():
            warped_impression, affine_matrix = model(template, impression)

        assert warped_impression.device.type == device.type, \
            f"Warped impression on wrong device: {warped_impression.device}"
        assert affine_matrix.device.type == device.type, \
            f"Affine matrix on wrong device: {affine_matrix.device}"
        assert warped_impression.device == template.device
        assert affine_matrix.device == template.device

    except Exception as e:
        pytest.fail(f"Model failed on device {device}: {e}")

def test_model_serialization():
    import tempfile
    import os
    model = GumNetAP(in_channels=1)
    template = torch.randn(2, 1, 192, 192)
    impression = torch.randn(2, 1, 192, 192)

    with torch.no_grad():
        orig_warped, orig_matrix = model(template, impression)

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        temp_path = f.name

    try:
        torch.save(model.state_dict(), temp_path)
        new_model = GumNetAP(in_channels=1)
        new_model.load_state_dict(torch.load(temp_path))

        with torch.no_grad():
            new_warped, new_matrix = new_model(template, impression)

        assert torch.allclose(orig_warped, new_warped, atol=1e-6), \
            "Serialization changed warped impression output"
        assert torch.allclose(orig_matrix, new_matrix, atol=1e-6), \
            "Serialization changed affine matrix output"

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# -----------------------------------------------------------------------------
# 5. EDGE CASE & ROBUSTNESS TESTS
# -----------------------------------------------------------------------------

def test_input_validation_shapes(gumnet_model):
    try:
        template_wrong = torch.randn(2, 1, 224, 224)
        impression = torch.randn(2, 1, 192, 192)
        
        with pytest.raises(RuntimeError):
            gumnet_model(template_wrong, impression)
            
    except AssertionError:
        pass

def test_zero_input(gumnet_model):
    template = torch.zeros(2, 1, 192, 192)
    impression = torch.zeros(2, 1, 192, 192)

    with torch.no_grad():
        warped_impression, affine_matrix = gumnet_model(template, impression)

    assert warped_impression.shape == (2, 1, 192, 192)
    assert affine_matrix.shape == (2, 2, 3)
    assert torch.isfinite(warped_impression).all()
    assert torch.isfinite(affine_matrix).all()

def test_extreme_values(gumnet_model):
    template = torch.randn(2, 1, 192, 192) * 1000
    impression = torch.randn(2, 1, 192, 192) * 1000

    with torch.no_grad():
        warped_impression, affine_matrix = gumnet_model(template, impression)

    assert torch.isfinite(warped_impression).all(), "Non-finite warped impression with large inputs"
    assert torch.isfinite(affine_matrix).all(), "Non-finite affine matrix with large inputs"

# -----------------------------------------------------------------------------
# 6. INTEGRATION TESTS
# -----------------------------------------------------------------------------

def test_end_to_end_pipeline():
    model = GumNetAP(in_channels=1)
    batch_size = 4
    templates = torch.randn(batch_size, 1, 192, 192)
    impressions = torch.randn(batch_size, 1, 192, 192)

    with torch.no_grad():
        warped_impressions, affine_matrices = model(templates, impressions)

    assert warped_impressions.shape == (batch_size, 1, 192, 192)
    assert affine_matrices.shape == (batch_size, 2, 3)
    assert torch.isfinite(warped_impressions).all()
    assert torch.isfinite(affine_matrices).all()
    
    differences = torch.abs(warped_impressions - impressions).mean(dim=[1, 2, 3])
    
    assert differences.mean() > 1e-6, "Warped impressions are identical to inputs"