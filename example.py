"""Example script demonstrating basic usage of RIS Research Engine."""

from ris_research_engine.foundation import SystemConfig, TrainingConfig, ExperimentConfig
from ris_research_engine.engine import ExperimentRunner


def main():
    """Run a quick test experiment."""
    
    print("=" * 60)
    print("RIS Research Engine - Quick Test Example")
    print("=" * 60)
    print()
    
    # Configure system
    system = SystemConfig(
        N=16,   # Small RIS array for quick test
        N_x=4,  # 4x4 grid
        N_y=4,
        K=16,   # Small codebook
        M=4     # Minimal sensing budget
    )
    
    # Configure training
    training = TrainingConfig(
        max_epochs=10,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=5
    )
    
    # Configure experiment
    config = ExperimentConfig(
        name="Quick Test",
        system=system,
        training=training,
        probe_type="random_uniform",
        probe_params={},
        model_type="mlp",
        model_params={},
        data_source="synthetic_rayleigh",
        data_params={'n_samples': 500},
        metrics=['top_1_accuracy', 'power_ratio']
    )
    
    # Run experiment
    print("Running experiment...")
    print(f"  Probe: {config.probe_type}")
    print(f"  Model: {config.model_type}")
    print(f"  System: N={system.N}, K={system.K}, M={system.M}")
    print()
    
    runner = ExperimentRunner()
    result = runner.run(config)
    
    # Display results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()
    print(f"Status: {result.status}")
    
    if result.status == 'completed':
        print()
        print("Metrics:")
        for metric_name, metric_value in result.metrics.items():
            print(f"  {metric_name}: {metric_value:.3f}")
        
        print()
        print("Training:")
        print(f"  Time: {result.training_time_seconds:.2f}s")
        print(f"  Epochs: {result.total_epochs}")
        print(f"  Best Epoch: {result.best_epoch}")
        print(f"  Parameters: {result.model_parameters:,}")
        
        # Display baseline comparison
        if result.baseline_results:
            print()
            print("Baseline Comparison:")
            ml_acc = result.metrics.get('top_1_accuracy', 0.0)
            print(f"  ML Model: {ml_acc:.3f}")
            
            for baseline_name, baseline_metrics in result.baseline_results.items():
                baseline_acc = baseline_metrics.get('top_1_accuracy', 0.0)
                improvement = ((ml_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
                print(f"  {baseline_name}: {baseline_acc:.3f} (improvement: {improvement:+.1f}%)")
        
        print()
        print("✓ Experiment completed successfully!")
    else:
        print(f"✗ Experiment failed: {result.error_message}")
    
    print()


if __name__ == '__main__':
    main()
