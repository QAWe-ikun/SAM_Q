"""
Configuration Parser and Management
====================================

Provides hierarchical configuration loading with inheritance support.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from copy import deepcopy


class Config:
    """
    Configuration manager with inheritance support.
    
    Allows configurations to inherit from base configs using _base_ key,
    enabling modular and reusable configuration files.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self._config: Dict[str, Any] = {}
        self._config_path: Optional[Path] = Path(config_path) if config_path else None
        
        if self._config_path:
            self.load(self._config_path)
    
    def load(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from YAML file with inheritance.
        
        Args:
            config_path: Path to YAML configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load base configuration if specified
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Handle inheritance
        if '_base_' in config:
            base_path = config_path.parent / config['_base_']
            base_config = Config(base_path)
            self._config = base_config.to_dict()
            del config['_base_']
        
        # Merge configurations
        self._deep_merge(self._config, config)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated key path (e.g., "model.qwen.hidden_dim")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated key path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return deepcopy(self._config)
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._config
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self._config})"
    
    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> None:
        """
        Recursively merge override dict into base dict.
        
        Args:
            base: Base dictionary
            override: Dictionary to merge
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_merge(base[key], value)
            else:
                base[key] = value


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    return Config(config_path).to_dict()
