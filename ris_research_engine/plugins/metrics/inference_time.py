"""Inference time metric."""

import torch
import time
from typing import Optional, Callable
from .base import BaseMetric


class InferenceTime(BaseMetric):
    """
    Model inference time metric.
    
    Measures the time taken for model inference in milliseconds.
    This metric should be computed differently - typically by timing
    the model forward pass rather than computing from predictions/targets.
    """
    
    name: str = "inference_time"
    description: str = "Model inference time in milliseconds"
    higher_is_better: bool = False  # Lower inference time is better
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None
    ) -> float:
        """
        Get inference time from metadata.
        
        Args:
            predictions: Model predictions (not used directly)
            targets: Ground truth targets (not used directly)
            metadata: Must contain 'inference_time_ms' value
            
        Returns:
            Inference time in milliseconds
            
        Raises:
            ValueError: If required metadata is missing
        """
        if metadata is None:
            raise ValueError(
                "InferenceTime metric requires metadata with 'inference_time_ms' key"
            )
        
        if 'inference_time_ms' not in metadata:
            raise ValueError("Metadata must contain 'inference_time_ms'")
        
        inference_time = metadata['inference_time_ms']
        
        if isinstance(inference_time, torch.Tensor):
            return inference_time.item()
        
        return float(inference_time)
    
    @staticmethod
    def measure_model_inference(
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        num_warmup: int = 10,
        num_iterations: int = 100,
        use_cuda: bool = None
    ) -> float:
        """
        Measure model inference time.
        
        Args:
            model: PyTorch model to measure
            input_tensor: Sample input tensor
            num_warmup: Number of warmup iterations (default: 10)
            num_iterations: Number of iterations to average (default: 100)
            use_cuda: Whether to use CUDA events for timing (auto-detect if None)
            
        Returns:
            Average inference time in milliseconds
        """
        model.eval()
        
        # Auto-detect CUDA
        if use_cuda is None:
            use_cuda = input_tensor.is_cuda
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(input_tensor)
        
        # Synchronize before timing
        if use_cuda:
            torch.cuda.synchronize()
        
        # Measure inference time
        if use_cuda:
            # Use CUDA events for precise GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(input_tensor)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            avg_time_ms = elapsed_time_ms / num_iterations
        else:
            # Use CPU timing
            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(input_tensor)
            end_time = time.perf_counter()
            
            elapsed_time_s = end_time - start_time
            avg_time_ms = (elapsed_time_s / num_iterations) * 1000
        
        return avg_time_ms


class AverageInferenceTime(InferenceTime):
    """Alias for InferenceTime with explicit averaging."""
    
    name: str = "average_inference_time"
    description: str = "Average model inference time in milliseconds"
