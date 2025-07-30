"""
Configuration Manager for EcoVision AI

Handles loading and managing configuration settings for the application.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration manager."""
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception:
                pass
        
        # Return default configuration
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "vision": {
                "model_type": "multi_scale_vit",
                "input_size": 224,
                "patch_size": 16,
                "batch_size": 32
            },
            "forecasting": {
                "sequence_length": 168,
                "prediction_horizon": 24,
                "tft_hidden_size": 256,
                "lstm_hidden_size": 128
            },
            "rl": {
                "learning_rate": 0.001,
                "epsilon": 0.1,
                "batch_size": 32,
                "memory_size": 100000
            },
            "data": {
                "satellite_api_key": "",
                "weather_api_key": ""
            },
            "monitoring": {
                "log_level": "INFO",
                "metrics_interval": 300
            },
            "alerts": {
                "deforestation_threshold": 0.05,
                "extreme_weather_threshold": 0.7
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports dot notation)."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save configuration to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)