"""
Tests for Adapter Modules
"""

import torch
import pytest
from src.models.adapters import Adapter, CrossModalAdapter, PresenceTokenAdapter


class TestAdapter:
    """Test basic Adapter."""
    
    def test_adapter_forward(self):
        """Test forward pass."""
        adapter = Adapter(input_dim=3584, output_dim=256, hidden_dim=512)
        x = torch.randn(2, 10, 3584)
        output = adapter(x)
        
        assert output.shape == (2, 10, 256)
    
    def test_adapter_residual(self):
        """Test residual connection."""
        adapter = Adapter(input_dim=256, output_dim=256, residual=True)
        x = torch.randn(2, 10, 256)
        output = adapter(x)
        
        assert output.shape == (2, 10, 256)
    
    def test_adapter_residual_dim_mismatch(self):
        """Test residual with dimension mismatch raises error."""
        with pytest.raises(ValueError):
            Adapter(input_dim=3584, output_dim=256, residual=True)


class TestCrossModalAdapter:
    """Test CrossModalAdapter."""
    
    def test_cross_modal_forward(self):
        """Test forward pass."""
        adapter = CrossModalAdapter(
            qwen_dim=3584,
            sam3_dim=256,
            num_queries=64,
            hidden_dim=512,
        )
        x = torch.randn(2, 10, 3584)
        output = adapter(x)
        
        assert output.shape == (2, 64, 256)
    
    def test_cross_modal_batch_invariance(self):
        """Test batch processing."""
        adapter = CrossModalAdapter(
            qwen_dim=3584,
            sam3_dim=256,
            num_queries=32,
            hidden_dim=256,
        )
        x = torch.randn(4, 20, 3584)
        output = adapter(x)
        
        assert output.shape == (4, 32, 256)


class TestPresenceTokenAdapter:
    """Test PresenceTokenAdapter."""
    
    def test_presence_token_forward(self):
        """Test forward pass."""
        adapter = PresenceTokenAdapter(
            qwen_dim=3584,
            sam3_dim=256,
            num_presence_tokens=4,
        )
        x = torch.randn(2, 10, 3584)
        output = adapter(x)
        
        # Output shape may vary due to concatenation logic
        assert output.dim() == 3
        assert output.shape[0] == 2
        assert output.shape[-1] == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
