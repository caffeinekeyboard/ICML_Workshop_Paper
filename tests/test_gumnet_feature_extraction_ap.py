import torch
import pytest

from model.gumnet_feature_extraction_ap import GumNetFeatureExtractionAP

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def feature_extractor():
    model = GumNetFeatureExtractionAP(in_channels=1)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 1. INITIALIZATION & SETUP TESTS
# -----------------------------------------------------------------------------

def test_model_initialization(feature_extractor):
    assert isinstance(feature_extractor, GumNetFeatureExtractionAP)
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
        output = feature_extractor(dummy_input, branch='Sa')

    expected_shape = (batch_size, 512, 19, 19)

    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {output.shape}"

def test_l2_normalization_enforcement(feature_extractor):
    dummy_input = torch.randn(2, 1, 192, 192)
    
    with torch.no_grad():
        output = feature_extractor(dummy_input, branch='Sa')

    l2_norms = torch.norm(output, p=2, dim=1)
    expected_norms = torch.ones_like(l2_norms)
    
    assert torch.allclose(l2_norms, expected_norms, atol=1e-6), \
        "The L2 norm along the channel dimension is not 1.0. FeatureL2Norm may be failing."

# -----------------------------------------------------------------------------
# 3. TRAINING & GRAPH INTEGRITY TESTS
# -----------------------------------------------------------------------------

def test_backward_pass_and_differentiability():
    model = GumNetFeatureExtractionAP(in_channels=1)
    model.train()
    dummy_input = torch.randn(2, 1, 192, 192)
    output = model(dummy_input, branch='Sa')
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
    model = GumNetFeatureExtractionAP(in_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = model.to(device)
        dummy_input = torch.randn(1, 1, 192, 192).to(device)
        with torch.no_grad():
            _ = model(dummy_input, branch='Sa')
    except Exception as e:
        pytest.fail(f"Model failed to execute after being moved to device {device}. Error: {e}")

def test_siamese_variable_batch_sizes(feature_extractor):
    batch_1 = torch.randn(1, 1, 192, 192)
    batch_4 = torch.randn(4, 1, 192, 192)

    with torch.no_grad():
        out_1 = feature_extractor(batch_1, branch='Sa')
        out_4 = feature_extractor(batch_4, branch='Sb')

    assert out_1.shape == (1, 512, 19, 19)
    assert out_4.shape == (4, 512, 19, 19)
    
# -----------------------------------------------------------------------------
# 5. ARCHITECTURAL ROUTING & PARAMETER SHARING TESTS
# -----------------------------------------------------------------------------

def test_branch_specific_batchnorm_gradients():
    model = GumNetFeatureExtractionAP(in_channels=1)
    model.train()
    dummy_input = torch.randn(2, 1, 192, 192)
    out_sa = model(dummy_input, branch='Sa')
    loss_sa = out_sa.mean()
    loss_sa.backward()

    assert model.bn1_sa.weight.grad is not None, "bn1_sa did not receive gradients."
    assert torch.abs(model.bn1_sa.weight.grad).sum() > 0, "bn1_sa gradients are zero."

    if model.bn1_sb.weight.grad is not None:
        assert torch.abs(model.bn1_sb.weight.grad).sum() == 0, \
            "bn1_sb received gradients during an 'Sa' forward pass. Branch isolation failed."

    model.zero_grad()
    out_sb = model(dummy_input, branch='Sb')
    loss_sb = out_sb.mean()
    loss_sb.backward()

    assert model.bn1_sb.weight.grad is not None, "bn1_sb did not receive gradients."
    assert torch.abs(model.bn1_sb.weight.grad).sum() > 0, "bn1_sb gradients are zero."

    if model.bn1_sa.weight.grad is not None:
        assert torch.abs(model.bn1_sa.weight.grad).sum() == 0, \
            "bn1_sa received gradients during an 'Sb' forward pass. Branch isolation failed."


def test_shared_convolution_gradient_accumulation():
    model = GumNetFeatureExtractionAP(in_channels=1)
    model.train()
    dummy_input = torch.randn(2, 1, 192, 192)
    model.zero_grad()
    out_sa = model(dummy_input, branch='Sa')
    out_sa.mean().backward()
    assert model.shared_conv1.weight.grad is not None
    grad_conv1_sa = model.shared_conv1.weight.grad.clone()
    model.zero_grad()
    out_sb = model(dummy_input, branch='Sb')
    out_sb.mean().backward()
    assert model.shared_conv1.weight.grad is not None
    grad_conv1_sb = model.shared_conv1.weight.grad.clone()
    model.zero_grad()
    out_sa_combined = model(dummy_input, branch='Sa')
    out_sb_combined = model(dummy_input, branch='Sb')
    combined_loss = out_sa_combined.mean() + out_sb_combined.mean()
    combined_loss.backward()
    assert model.shared_conv1.weight.grad is not None
    grad_conv1_combined = model.shared_conv1.weight.grad.clone()
    expected_combined_grad = grad_conv1_sa + grad_conv1_sb

    assert torch.allclose(grad_conv1_combined, expected_combined_grad, atol=1e-5), \
        "Convolutional layers are not correctly sharing gradients across branches."


def test_forward_pass_batchnorm_isolation():
    model = GumNetFeatureExtractionAP(in_channels=1)
    model.eval()
    dummy_input = torch.randn(4, 1, 192, 192)
    with torch.no_grad():
        assert model.bn1_sa.running_mean is not None
        model.bn1_sa.running_mean.fill_(100.0)
        assert model.bn1_sa.running_var is not None
        model.bn1_sa.running_var.fill_(0.1)
        assert model.bn1_sb.running_mean is not None
        model.bn1_sb.running_mean.fill_(-100.0)
        assert model.bn1_sb.running_var is not None
        model.bn1_sb.running_var.fill_(0.1)
        out_sa = model(dummy_input, branch='Sa')
        out_sb = model(dummy_input, branch='Sb')

    assert not torch.allclose(out_sa, out_sb, atol=1e-3), \
        "Outputs are identical despite drastically different BN parameters. " \
        "The forward pass is likely ignoring the branch-specific routing."