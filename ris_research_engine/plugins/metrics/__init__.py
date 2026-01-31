"""Metric plugin system with auto-discovery."""

from .base import BaseMetric
from .top_k_accuracy import TopKAccuracy, Top1Accuracy, Top5Accuracy, Top10Accuracy
from .power_ratio import PowerRatio
from .hit_at_l import HitAtL, HitAt1, HitAt5, HitAt10
from .mean_reciprocal_rank import MeanReciprocalRank, MRR
from .spectral_efficiency import SpectralEfficiency, AverageSpectralEfficiency
from .inference_time import InferenceTime, AverageInferenceTime


# Auto-import all metrics
_METRICS = {}


def register_metric(metric_class):
    """Register a metric class in the registry."""
    _METRICS[metric_class.name] = metric_class


def get_metric(name: str, **kwargs) -> BaseMetric:
    """
    Get a metric instance by name.
    
    Args:
        name: Name of the metric class
        **kwargs: Additional arguments to pass to metric constructor
        
    Returns:
        Instance of the requested metric class
        
    Raises:
        KeyError: If metric name is not registered
    """
    if name not in _METRICS:
        raise KeyError(
            f"Metric '{name}' not found. Available metrics: {list_metrics()}"
        )
    return _METRICS[name](**kwargs)


def list_metrics():
    """List all registered metric names."""
    return list(_METRICS.keys())


# Register all built-in metrics
register_metric(TopKAccuracy)
register_metric(Top1Accuracy)
register_metric(Top5Accuracy)
register_metric(Top10Accuracy)
register_metric(PowerRatio)
register_metric(HitAtL)
register_metric(HitAt1)
register_metric(HitAt5)
register_metric(HitAt10)
register_metric(MeanReciprocalRank)
register_metric(MRR)
register_metric(SpectralEfficiency)
register_metric(AverageSpectralEfficiency)
register_metric(InferenceTime)
register_metric(AverageInferenceTime)


__all__ = [
    'BaseMetric',
    'TopKAccuracy',
    'Top1Accuracy',
    'Top5Accuracy',
    'Top10Accuracy',
    'PowerRatio',
    'HitAtL',
    'HitAt1',
    'HitAt5',
    'HitAt10',
    'MeanReciprocalRank',
    'MRR',
    'SpectralEfficiency',
    'AverageSpectralEfficiency',
    'InferenceTime',
    'AverageInferenceTime',
    'register_metric',
    'get_metric',
    'list_metrics',
]
