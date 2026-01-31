#!/usr/bin/env python3
"""
Example script demonstrating the RIS Auto-Research Engine.

This script shows basic usage of the engine without requiring a full Jupyter environment.
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Run a simple example experiment."""
    
    print("="*70)
    print("RIS Auto-Research Engine - Example Script")
    print("="*70)
    print()
    
    try:
        # Import the engine
        from ris_research_engine import RISEngine
        
        print("✓ Successfully imported RISEngine")
        print()
        
        # Create engine instance
        engine = RISEngine(db_path="example.db", output_dir="example_outputs")
        print("✓ Created engine instance")
        print()
        
        # Show available plugins
        print("Available plugins:")
        print(f"  Probes: {', '.join(engine.experiment_runner._probes[:5])}...")
        print(f"  Models: {', '.join(engine.experiment_runner._models[:5])}...")
        print()
        
        # Example 1: Run a quick experiment
        print("Example 1: Running a quick experiment")
        print("-" * 70)
        
        result = engine.run(
            probe='hadamard',
            model='mlp',
            M=8,
            K=64,
            N=64,
            epochs=10,
            n_samples=1000,
            learning_rate=1e-3,
            batch_size=32
        )
        
        print(f"Experiment completed!")
        print(f"  Status: {result.status}")
        print(f"  Training time: {result.training_time_seconds:.2f}s")
        print(f"  Test accuracy: {result.metrics.get('top_1_accuracy', 0.0):.4f}")
        print()
        
        # Example 2: Show history
        print("Example 2: Viewing experiment history")
        print("-" * 70)
        
        history = engine.show_history(limit=5)
        print(f"Found {len(history)} experiments in database")
        if not history.empty:
            print(history.to_string(index=False))
        print()
        
        # Example 3: Run a quick search campaign
        print("Example 3: Running automated search (using quick_test config)")
        print("-" * 70)
        print("This will compare 2 probes with MLP model...")
        
        campaign = engine.search(config_path='configs/search_spaces/quick_test.yaml')
        
        print(f"Search campaign completed!")
        print(f"  Campaign: {campaign.campaign_name}")
        print(f"  Total experiments: {campaign.total_experiments}")
        print(f"  Completed: {campaign.completed_experiments}")
        print(f"  Failed: {campaign.failed_experiments}")
        if campaign.best_result:
            print(f"  Best accuracy: {campaign.best_result.primary_metric_value:.4f}")
            print(f"  Best config: {campaign.best_result.config.probe_type} + {campaign.best_result.config.model_type}")
        print()
        
        print("="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        print()
        print("Next steps:")
        print("  - Check 'example_outputs/' for generated plots")
        print("  - Browse 'example.db' for stored results")
        print("  - Try notebooks/01_quickstart.ipynb for interactive usage")
        print("  - Run: python -m ris_research_engine.ui.cli --help")
        
        return 0
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print()
        print("Please install dependencies:")
        print("  pip install -e .")
        return 1
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
