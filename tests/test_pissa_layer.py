"""
Unit tests for PiSSA layer.
"""
import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pissa import PiSSALayer


def test_pissa_layer_creation():
    """Test PiSSA layer creation."""
    layer = PiSSALayer(in_features=128, out_features=256, r=8)
    
    assert layer.in_features == 128
    assert layer.out_features == 256
    assert layer.r == 8
    assert layer.pissa_A.shape == (8, 128)
    assert layer.pissa_B.shape == (256, 8)


def test_pissa_layer_forward():
    """Test PiSSA layer forward pass."""
    layer = PiSSALayer(in_features=128, out_features=256, r=8)
    x = torch.randn(32, 128)
    
    output = layer(x)
    
    assert output.shape == (32, 256)


def test_pissa_layer_initialization():
    """Test PiSSA layer initialization from pretrained weights."""
    # Create a pretrained linear layer
    linear = nn.Linear(128, 256, bias=True)
    
    # Create PiSSA layer and initialize from pretrained
    pissa_layer = PiSSALayer(in_features=128, out_features=256, r=8)
    pissa_layer.initialize_from_weight(linear.weight.data)
    
    # Test that the reconstruction is close to original
    x = torch.randn(32, 128)
    original_output = linear(x)
    pissa_output = pissa_layer(x)
    
    # The reconstruction should be close (not exact due to low-rank approximation)
    reconstruction_error = torch.norm(original_output - pissa_output) / torch.norm(original_output)
    
    # With r=8 for 128x256 matrix, we expect some error but not too large
    assert reconstruction_error < 0.5, f"Reconstruction error too large: {reconstruction_error}"


def test_pissa_trainable_parameters():
    """Test that only principal components are trainable."""
    layer = PiSSALayer(in_features=128, out_features=256, r=8)
    
    # Principal components should be trainable
    assert layer.pissa_A.requires_grad
    assert layer.pissa_B.requires_grad
    
    # Residual should not be trainable (it's a buffer)
    assert not layer.residual_W.requires_grad


def test_pissa_layer_shapes():
    """Test various input shapes."""
    layer = PiSSALayer(in_features=64, out_features=32, r=4)
    
    # 2D input
    x1 = torch.randn(16, 64)
    out1 = layer(x1)
    assert out1.shape == (16, 32)
    
    # 3D input (batched sequences)
    x2 = torch.randn(8, 10, 64)
    out2 = layer(x2)
    assert out2.shape == (8, 10, 32)


if __name__ == "__main__":
    # Run tests
    test_pissa_layer_creation()
    print("✓ test_pissa_layer_creation passed")
    
    test_pissa_layer_forward()
    print("✓ test_pissa_layer_forward passed")
    
    test_pissa_layer_initialization()
    print("✓ test_pissa_layer_initialization passed")
    
    test_pissa_trainable_parameters()
    print("✓ test_pissa_trainable_parameters passed")
    
    test_pissa_layer_shapes()
    print("✓ test_pissa_layer_shapes passed")
    
    print("\nAll tests passed!")
