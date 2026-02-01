"""Test all metrics: perfect=1.0, random≈1/K, power_ratio in [0,1]."""

import pytest
import torch
import numpy as np

from ris_research_engine.plugins.metrics import (
    get_metric, list_metrics,
    TopKAccuracy, Top1Accuracy, PowerRatio,
    HitAtL, MeanReciprocalRank, SpectralEfficiency
)


# Main 6 metric types
METRIC_TYPES = [
    "top_k_accuracy",
    "top_1_accuracy", 
    "power_ratio",
    "hit_at_l",
    "mean_reciprocal_rank",
    "spectral_efficiency"
]


class TestMetricRegistry:
    """Test metric registration and discovery."""
    
    def test_list_metrics(self):
        """Test that metrics are registered."""
        metrics = list_metrics()
        assert len(metrics) >= 6
        for metric_type in METRIC_TYPES:
            assert metric_type in metrics
    
    def test_get_metric(self):
        """Test getting metric instances."""
        for metric_type in METRIC_TYPES:
            metric = get_metric(metric_type)
            assert metric is not None
            assert metric.name == metric_type
    
    def test_get_invalid_metric(self):
        """Test that invalid metric name raises error."""
        with pytest.raises(KeyError):
            get_metric("nonexistent_metric")


class TestPerfectScores:
    """Test that perfect predictions give score of 1.0."""
    
    def test_top1_perfect(self, perfect_predictions, sample_labels):
        """Test Top-1 accuracy = 1.0 for perfect predictions."""
        metric = get_metric("top_1_accuracy")
        
        # Create perfect predictions: highest score at correct index
        batch_size, N = perfect_predictions.shape
        preds = torch.zeros_like(perfect_predictions)
        for i, label in enumerate(sample_labels):
            preds[i, label] = 10.0  # Highest score at correct position
        
        score = metric.compute(preds, sample_labels)
        assert abs(score - 1.0) < 1e-6, f"Perfect prediction should give 1.0, got {score}"
    
    def test_topk_perfect(self, sample_labels):
        """Test Top-K accuracy = 1.0 for perfect predictions."""
        metric = get_metric("top_k_accuracy", k=5)
        
        batch_size = len(sample_labels)
        N = 16
        preds = torch.randn(batch_size, N)
        
        # Ensure correct label is in top-k
        for i, label in enumerate(sample_labels):
            preds[i, label] = 100.0
        
        score = metric.compute(preds, sample_labels)
        assert abs(score - 1.0) < 1e-6
    
    def test_mrr_perfect(self, sample_labels):
        """Test MRR = 1.0 for perfect predictions."""
        metric = get_metric("mean_reciprocal_rank")
        
        batch_size = len(sample_labels)
        N = 16
        preds = torch.randn(batch_size, N)
        
        # Put correct label at top
        for i, label in enumerate(sample_labels):
            preds[i, label] = 100.0
        
        score = metric.compute(preds, sample_labels)
        assert abs(score - 1.0) < 1e-6, f"Perfect MRR should be 1.0, got {score}"
    
    def test_hit_at_l_perfect(self, sample_labels):
        """Test Hit@L = 1.0 for perfect predictions."""
        metric = get_metric("hit_at_l", L=5)
        
        batch_size = len(sample_labels)
        N = 16
        preds = torch.randn(batch_size, N)
        
        # Ensure correct label is in top-L
        for i, label in enumerate(sample_labels):
            preds[i, label] = 100.0
        
        score = metric.compute(preds, sample_labels)
        assert abs(score - 1.0) < 1e-6


class TestRandomScores:
    """Test that random predictions give score ≈ 1/K."""
    
    def test_top1_random(self, seed):
        """Test Top-1 accuracy ≈ 1/N for random predictions."""
        metric = get_metric("top_1_accuracy")
        
        N = 16
        batch_size = 1000  # Large batch for stable statistics
        
        # Random predictions
        preds = torch.randn(batch_size, N)
        labels = torch.randint(0, N, (batch_size,))
        
        score = metric.compute(preds, labels)
        
        # Should be close to 1/N for random
        expected = 1.0 / N
        assert abs(score - expected) < 0.05, f"Random should be ~{expected}, got {score}"
    
    def test_topk_random(self, seed):
        """Test Top-K accuracy ≈ K/N for random predictions."""
        K = 5
        N = 16
        metric = get_metric("top_k_accuracy", k=K)
        
        batch_size = 1000
        preds = torch.randn(batch_size, N)
        labels = torch.randint(0, N, (batch_size,))
        
        score = metric.compute(preds, labels)
        
        expected = K / N
        assert abs(score - expected) < 0.1, f"Random should be ~{expected}, got {score}"
    
    def test_hit_random(self, seed):
        """Test Hit@L ≈ L/N for random predictions."""
        L = 5
        N = 16
        metric = get_metric("hit_at_l", L=L)
        
        batch_size = 1000
        preds = torch.randn(batch_size, N)
        labels = torch.randint(0, N, (batch_size,))
        
        score = metric.compute(preds, labels)
        
        expected = L / N
        assert abs(score - expected) < 0.1, f"Random should be ~{expected}, got {score}"


class TestPowerRatio:
    """Test power ratio metric is in [0, 1] range."""
    
    def test_power_ratio_range(self):
        """Test power ratio is in [0, 1]."""
        metric = get_metric("power_ratio")
        
        # Generate random channel and configuration
        K, N = 16, 16
        H = torch.randn(K, N, dtype=torch.complex64)
        theta = torch.rand(N) * 2 * np.pi
        
        power = metric.compute(H, theta)
        
        assert power >= 0.0, f"Power ratio should be >= 0, got {power}"
        assert power <= 1.0, f"Power ratio should be <= 1, got {power}"
    
    def test_power_ratio_optimal(self):
        """Test power ratio with optimal configuration."""
        metric = get_metric("power_ratio")
        
        K, N = 8, 8
        # Simple channel
        H = torch.ones(K, N, dtype=torch.complex64)
        # Aligned phases
        theta = torch.zeros(N)
        
        power = metric.compute(H, theta)
        
        # Should be high for aligned configuration
        assert power >= 0.0
        assert power <= 1.0
    
    def test_power_ratio_random_configs(self, seed):
        """Test power ratio for multiple random configurations."""
        metric = get_metric("power_ratio")
        
        K, N = 16, 16
        
        for _ in range(10):
            H = torch.randn(K, N, dtype=torch.complex64)
            theta = torch.rand(N) * 2 * np.pi
            
            power = metric.compute(H, theta)
            assert 0.0 <= power <= 1.0, f"Power ratio out of range: {power}"


class TestSpectralEfficiency:
    """Test spectral efficiency metric."""
    
    def test_spectral_efficiency_positive(self):
        """Test spectral efficiency is positive."""
        metric = get_metric("spectral_efficiency")
        
        K, N = 16, 16
        H = torch.randn(K, N, dtype=torch.complex64)
        theta = torch.rand(N) * 2 * np.pi
        snr_db = 20.0
        
        se = metric.compute(H, theta, snr_db)
        
        assert se >= 0.0, f"Spectral efficiency should be positive, got {se}"
    
    def test_spectral_efficiency_increases_with_snr(self):
        """Test SE increases with SNR."""
        metric = get_metric("spectral_efficiency")
        
        K, N = 16, 16
        H = torch.randn(K, N, dtype=torch.complex64)
        theta = torch.rand(N) * 2 * np.pi
        
        se_low = metric.compute(H, theta, snr_db=0.0)
        se_high = metric.compute(H, theta, snr_db=20.0)
        
        assert se_high >= se_low, "SE should increase with SNR"
    
    def test_spectral_efficiency_finite(self):
        """Test SE produces finite values."""
        metric = get_metric("spectral_efficiency")
        
        K, N = 16, 16
        
        for _ in range(10):
            H = torch.randn(K, N, dtype=torch.complex64)
            theta = torch.rand(N) * 2 * np.pi
            
            se = metric.compute(H, theta, snr_db=10.0)
            assert np.isfinite(se), f"SE should be finite, got {se}"


class TestMetricBehavior:
    """Test metric behavior and edge cases."""
    
    def test_top1_all_wrong(self):
        """Test Top-1 = 0 when all predictions are wrong."""
        metric = get_metric("top_1_accuracy")
        
        batch_size, N = 8, 16
        
        # Create predictions that are always wrong
        preds = torch.zeros(batch_size, N)
        labels = torch.zeros(batch_size, dtype=torch.long)
        
        # Make sure wrong index has highest score
        for i in range(batch_size):
            preds[i, (labels[i] + 1) % N] = 1.0
        
        score = metric.compute(preds, labels)
        assert score == 0.0, "All wrong predictions should give 0.0"
    
    def test_mrr_worst_case(self):
        """Test MRR with worst ranking."""
        metric = get_metric("mean_reciprocal_rank")
        
        batch_size, N = 8, 16
        
        preds = torch.zeros(batch_size, N)
        labels = torch.zeros(batch_size, dtype=torch.long)
        
        # Put correct label at last position
        for i in range(batch_size):
            preds[i, :] = torch.arange(N, 0, -1, dtype=torch.float)
            labels[i] = 0  # Correct label at position with lowest score
        
        score = metric.compute(preds, labels)
        
        # MRR should be 1/N for last position
        expected = 1.0 / N
        assert abs(score - expected) < 0.01
    
    def test_metric_batch_consistency(self):
        """Test metrics are consistent across batch sizes."""
        metric = get_metric("top_1_accuracy")
        
        N = 16
        
        # Single sample
        preds_single = torch.randn(1, N)
        labels_single = torch.tensor([0])
        score_single = metric.compute(preds_single, labels_single)
        
        # Replicate as batch
        preds_batch = preds_single.repeat(10, 1)
        labels_batch = labels_single.repeat(10)
        score_batch = metric.compute(preds_batch, labels_batch)
        
        assert abs(score_single - score_batch) < 1e-6, "Metric should be consistent across batch sizes"


class TestMetricParameters:
    """Test metric parameter handling."""
    
    def test_topk_different_k(self):
        """Test Top-K with different K values."""
        batch_size, N = 8, 16
        preds = torch.randn(batch_size, N)
        labels = torch.randint(0, N, (batch_size,))
        
        scores = []
        for k in [1, 3, 5, 10]:
            metric = get_metric("top_k_accuracy", k=k)
            score = metric.compute(preds, labels)
            scores.append(score)
        
        # Top-K accuracy should increase with K
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i+1], "Top-K should increase with K"
    
    def test_hit_at_different_l(self):
        """Test Hit@L with different L values."""
        batch_size, N = 8, 16
        preds = torch.randn(batch_size, N)
        labels = torch.randint(0, N, (batch_size,))
        
        scores = []
        for L in [1, 3, 5, 10]:
            metric = get_metric("hit_at_l", L=L)
            score = metric.compute(preds, labels)
            scores.append(score)
        
        # Hit@L should increase with L
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i+1], "Hit@L should increase with L"
