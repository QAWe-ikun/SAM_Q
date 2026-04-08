"""
Tests for Configuration Parser
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from src.utils.config import Config, load_config


class TestConfig:
    """Test Config class."""
    
    def test_config_load(self):
        """Test loading from YAML file."""
        config_data = {
            "model": {
                "qwen": {
                    "hidden_dim": 3584,
                },
            },
            "training": {
                "num_epochs": 100,
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        config = Config(config_path)
        
        assert config.get("model.qwen.hidden_dim") == 3584
        assert config.get("training.num_epochs") == 100
        
        # Cleanup
        Path(config_path).unlink()
    
    def test_config_inheritance(self):
        """Test configuration inheritance."""
        base_config = {
            "model": {"lr": 1e-4},
            "training": {"batch_size": 4},
        }
        
        override_config = {
            "_base_": "base.yaml",
            "training": {"batch_size": 8},
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write base config
            base_path = Path(tmpdir) / "base.yaml"
            with open(base_path, 'w') as f:
                yaml.dump(base_config, f)
            
            # Write override config
            override_path = Path(tmpdir) / "override.yaml"
            with open(override_path, 'w') as f:
                yaml.dump(override_config, f)
            
            # Load override
            config = Config(override_path)
            
            # Should inherit base model.lr
            assert config.get("model.lr") == 1e-4
            # Should override training.batch_size
            assert config.get("training.batch_size") == 8
    
    def test_config_get_with_default(self):
        """Test get with default value."""
        config = Config()
        config._config = {"a": {"b": 1}}
        
        assert config.get("a.b") == 1
        assert config.get("a.c", default=42) == 42
        assert config.get("x.y.z", default="default") == "default"
    
    def test_config_set(self):
        """Test set value."""
        config = Config()
        config.set("model.qwen.hidden_dim", 3584)
        
        assert config.get("model.qwen.hidden_dim") == 3584
    
    def test_load_config_function(self):
        """Test load_config convenience function."""
        config_data = {"key": "value"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        result = load_config(config_path)
        
        assert result["key"] == "value"
        
        # Cleanup
        Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
