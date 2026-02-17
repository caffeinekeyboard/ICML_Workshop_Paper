import torch
import pytest

from model.gumnet_siamese_matching_mp import GumNetSiameseMatchingMP

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def siamese_matcher():
    model = GumNetSiameseMatchingMP(in_channels=361, out_channels=1024)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 1. INITIALIZATION & SETUP TESTS
# -----------------------------------------------------------------------------

def test_model_initialization(siamese_matcher):
    assert isinstance(siamese_matcher, GumNetSiameseMatchingMP)
    assert hasattr(siamese_matcher, 'correlation_layer')
    assert hasattr(siamese_matcher, 'regression_block_ab')
    assert hasattr(siamese_matcher, 'regression_block_ba')
    assert hasattr(siamese_matcher, 'l2_norm')

def test_pseudo_siamese_unshared_weights(siamese_matcher):
    ab_weight = siamese_matcher.regression_block_ab[0].weight
    ba_weight = siamese_matcher.regression_block_ba[0].weight

    assert id(ab_weight) != id(ba_weight), \
        "Weights are shared! The pseudo-siamese architecture is compromised."
    assert not torch.allclose(ab_weight, ba_weight), \
        "Weights are identical across branches upon initialization."

# -----------------------------------------------------------------------------
# 2. FUNCTIONAL TESTS (DIMENSIONALITY & CORRECTNESS)
# -----------------------------------------------------------------------------

def test_forward_pass_dimensions(siamese_matcher):
    batch_size = 2
    channels = 512
    spatial = 19
    v_a = torch.randn(batch_size, channels, spatial, spatial)
    v_b = torch.randn(batch_size, channels, spatial, spatial)
    
    with torch.no_grad():
        out_ab, out_ba = siamese_matcher(v_a, v_b)

    expected_shape = (batch_size, 50176)

    assert out_ab.shape == expected_shape, \
        f"Expected out_ab shape {expected_shape}, but got {out_ab.shape}"   
    assert out_ba.shape == expected_shape, \
        f"Expected out_ba shape {expected_shape}, but got {out_ba.shape}"

# -----------------------------------------------------------------------------
# 3. TRAINING & GRAPH INTEGRITY TESTS
# -----------------------------------------------------------------------------

def test_backward_pass_and_differentiability():
    model = GumNetSiameseMatchingMP(in_channels=361, out_channels=1024)
    model.train()
    v_a = torch.randn(2, 512, 19, 19, requires_grad=True)
    v_b = torch.randn(2, 512, 19, 19, requires_grad=True)
    out_ab, out_ba = model(v_a, v_b)
    dummy_loss = out_ab.sum() + out_ba.sum()
    dummy_loss.backward()

    assert v_a.grad is not None, "Gradients did not reach input v_a."
    assert v_b.grad is not None, "Gradients did not reach input v_b."

    ab_grad = model.regression_block_ab[0].weight.grad
    ba_grad = model.regression_block_ba[0].weight.grad

    assert ab_grad is not None and torch.sum(torch.abs(ab_grad)) > 0, \
        "Gradients failed to update regression_block_ab."
    assert ba_grad is not None and torch.sum(torch.abs(ba_grad)) > 0, \
        "Gradients failed to update regression_block_ba."

# -----------------------------------------------------------------------------
# 4. HARDWARE & ECOSYSTEM COMPATIBILITY TESTS
# -----------------------------------------------------------------------------

def test_device_agnosticism():
    model = GumNetSiameseMatchingMP(in_channels=361, out_channels=32) # Smaller channels for VRAM safety in tests
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = model.to(device)
        v_a = torch.randn(2, 512, 19, 19).to(device)
        v_b = torch.randn(2, 512, 19, 19).to(device)
        
        with torch.no_grad():
            out_ab, out_ba = model(v_a, v_b)
            
        # FIX: Check .device.type to safely ignore the 'cuda:0' vs 'cuda' index mismatch
        assert out_ab.device.type == device.type, "Output out_ab did not remain on the target device."
        assert out_ba.device.type == device.type, "Output out_ba did not remain on the target device."
        
        # Alternatively, checking against the input tensor's exact device is also foolproof:
        assert out_ab.device == v_a.device, "Output out_ab jumped to a different device."
        
    except Exception as e:
        pytest.fail(f"Model failed to execute after being moved to device {device}. Error: {e}")

def test_siamese_variable_batch_sizes(siamese_matcher):
    batch_1_a = torch.randn(1, 512, 19, 19)
    batch_1_b = torch.randn(1, 512, 19, 19)
    batch_4_a = torch.randn(4, 512, 19, 19)
    batch_4_b = torch.randn(4, 512, 19, 19)

    with torch.no_grad():
        out_ab_1, _ = siamese_matcher(batch_1_a, batch_1_b)
        out_ab_4, _ = siamese_matcher(batch_4_a, batch_4_b)

    assert out_ab_1.shape == (1, 50176), f"Failed on Batch Size 1: {out_ab_1.shape}"
    assert out_ab_4.shape == (4, 50176), f"Failed on Batch Size 4: {out_ab_4.shape}"