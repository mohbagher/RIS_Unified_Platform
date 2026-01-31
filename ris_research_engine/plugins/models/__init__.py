"""Model plugin system with auto-discovery."""

from .base import BaseModel
from .mlp import MLPModel
from .residual_mlp import ResidualMLPModel
from .cnn_1d import CNN1DModel
from .cnn_2d import CNN2DModel
from .transformer import TransformerModel
from .set_transformer import SetTransformerModel
from .lstm import LSTMModel


# Auto-import all models
_MODELS = {}


def register_model(model_class):
    """Register a model class in the registry."""
    _MODELS[model_class.name] = model_class


def get_model(name: str) -> BaseModel:
    """
    Get a model instance by name.
    
    Args:
        name: Name of the model class
        
    Returns:
        Instance of the requested model class
        
    Raises:
        KeyError: If model name is not registered
    """
    if name not in _MODELS:
        raise KeyError(
            f"Model '{name}' not found. Available models: {list_models()}"
        )
    return _MODELS[name]()


def list_models():
    """List all registered model names."""
    return list(_MODELS.keys())


# Register all built-in models
register_model(MLPModel)
register_model(ResidualMLPModel)
register_model(CNN1DModel)
register_model(CNN2DModel)
register_model(TransformerModel)
register_model(SetTransformerModel)
register_model(LSTMModel)


__all__ = [
    'BaseModel',
    'MLPModel',
    'ResidualMLPModel',
    'CNN1DModel',
    'CNN2DModel',
    'TransformerModel',
    'SetTransformerModel',
    'LSTMModel',
    'register_model',
    'get_model',
    'list_models',
]
