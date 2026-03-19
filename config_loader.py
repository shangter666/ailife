import os
import yaml
from pathlib import Path
from pydantic import BaseModel

class LLMSettings(BaseModel):
    api_key: str
    base_url: str
    model_name: str
    temperature: float

class StorageSettings(BaseModel):
    memory_path: str

class AppConfig(BaseModel):
    llm_settings: LLMSettings
    storage_settings: StorageSettings

def load_config(config_path: str = "config.yaml") -> AppConfig:
    """
    Load configuration from a YAML file.
    If the path is relative, it is resolved relative to the directory of this file.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).parent / config_path
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
        
    return AppConfig(**config_dict)

# Global configuration instance
config = load_config()

# Ensures that anyone importing `config` from `config_loader` gets the loaded config.
# Example: 
# from config_loader import config
# print(config.llm_settings.model_name)
