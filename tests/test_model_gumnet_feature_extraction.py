import torch
import pytest

from model.gumnet_feature_extraction import GumNetFeatureExtraction

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def feature_extractor():
    model = GumNetFeatureExtraction(in_channels=1)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 1. INITIALIZATION & SETUP TESTS
# -----------------------------------------------------------------------------

def test_model_initialization(feature_extractor):
    assert isinstance(feature_extractor, GumNetFeatureExtraction)
    assert hasattr(feature_extractor, 'shared_conv1')
    assert hasattr(feature_extractor, 'pool4')
    assert hasattr(feature_extractor, 'l2_norm')

# -----------------------------------------------------------------------------
# 2. FUNCTIONAL TESTS (DIMENSIONALITY & CORRECTNESS)
# -----------------------------------------------------------------------------

def test_forward_pass_dimensions(feature_extractor):
    batch_size = 2
    in_channels = 1
    dummy_input = torch.randn(batch_size, in_channels, 192, 192)
    
    with torch.no_grad():
        output = feature_extractor(dummy_input)

    expected_shape = (batch_size, 512, 14, 14)

    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {output.shape}"

def test_l2_normalization_enforcement(feature_extractor):
    dummy_input = torch.randn(2, 1, 192, 192)
    
    with torch.no_grad():
        output = feature_extractor(dummy_input)

    l2_norms = torch.norm(output, p=2, dim=1)
    expected_norms = torch.ones_like(l2_norms)
    
    assert torch.allclose(l2_norms, expected_norms, atol=1e-6), \
        "The L2 norm along the channel dimension is not 1.0. FeatureL2Norm may be failing."

# -----------------------------------------------------------------------------
# 3. TRAINING & GRAPH INTEGRITY TESTS
# -----------------------------------------------------------------------------

def test_backward_pass_and_differentiability():
    model = GumNetFeatureExtraction(in_channels=1)
    model.train()
    dummy_input = torch.randn(2, 1, 192, 192)
    output = model(dummy_input)
    dummy_loss = output.mean()
    dummy_loss.backward()

    assert model.shared_conv1.weight.grad is not None, \
        "Gradients did not reach the first layer. The computational graph is broken."

    assert torch.sum(torch.abs(model.shared_conv1.weight.grad)) > 0, \
        "Gradients reached the first layer but are entirely zero."

# -----------------------------------------------------------------------------
# 4. HARDWARE & ECOSYSTEM COMPATIBILITY TESTS
# -----------------------------------------------------------------------------

def test_device_agnosticism():
    model = GumNetFeatureExtraction(in_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = model.to(device)
        dummy_input = torch.randn(1, 1, 192, 192).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
        pytest.fail(f"Model failed to execute after being moved to device {device}. Error: {e}")

def test_siamese_variable_batch_sizes(feature_extractor):
    batch_1 = torch.randn(1, 1, 192, 192)
    batch_4 = torch.randn(4, 1, 192, 192)

    with torch.no_grad():
        out_1 = feature_extractor(batch_1)
        out_4 = feature_extractor(batch_4)

    assert out_1.shape == (1, 512, 14, 14)
    assert out_4.shape == (4, 512, 14, 14)