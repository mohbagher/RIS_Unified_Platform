"""Basic smoke tests for the RIS research engine."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Foundation
        from ris_research_engine.foundation import data_types, storage
        
        # Plugins
        from ris_research_engine.plugins import probes, models, metrics, data_sources, baselines, search
        
        # Engine - may fail if torch not installed
        try:
            from ris_research_engine import engine
        except (ImportError, OSError) as e:
            print(f"  ⚠ Engine import skipped (torch not available): {e.__class__.__name__}")
        
        # UI
        try:
            from ris_research_engine import ui
        except (ImportError, OSError) as e:
            print(f"  ⚠ UI import skipped (dependencies not available): {e.__class__.__name__}")
        
        print("✓ Core imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        raise


def test_plugin_discovery():
    """Test that plugins are discovered correctly."""
    print("\nTesting plugin discovery...")
    
    from ris_research_engine.plugins.probes import list_probes
    from ris_research_engine.plugins.models import list_models
    from ris_research_engine.plugins.metrics import list_metrics
    from ris_research_engine.plugins.data_sources import list_data_sources
    from ris_research_engine.plugins.baselines import AVAILABLE_BASELINES
    from ris_research_engine.plugins.search import list_strategies
    
    probes = list_probes()
    models = list_models()
    metrics = list_metrics()
    data_sources = list_data_sources()
    baselines = list(AVAILABLE_BASELINES.keys())
    strategies = list(list_strategies().keys())
    
    print(f"  Probes: {len(probes)} found - {probes[:3]}...")
    print(f"  Models: {len(models)} found - {models[:3]}...")
    print(f"  Metrics: {len(metrics)} found - {metrics[:3]}...")
    print(f"  Data sources: {len(data_sources)} found")
    print(f"  Baselines: {len(baselines)} found")
    print(f"  Search strategies: {len(strategies)} found")
    
    assert len(probes) >= 6, "Should have at least 6 probes"
    assert len(models) >= 6, "Should have at least 6 models"
    assert len(metrics) >= 10, "Should have at least 10 metrics"
    
    print("✓ Plugin discovery successful")


def test_data_types():
    """Test data type creation and validation."""
    print("\nTesting data types...")
    
    from ris_research_engine.foundation.data_types import SystemConfig, TrainingConfig
    
    # Create configs
    system = SystemConfig(N=64, N_x=8, N_y=8, K=64, M=8)
    training = TrainingConfig(max_epochs=10)
    
    # Validate
    assert system.validate()
    assert training.validate()
    
    # Test serialization
    system_dict = system.to_dict()
    system_restored = SystemConfig.from_dict(system_dict)
    assert system_restored.N == 64
    
    print("✓ Data types working correctly")


def test_result_tracker():
    """Test result tracker with in-memory database."""
    print("\nTesting result tracker...")
    
    from ris_research_engine.foundation.storage import ResultTracker
    from ris_research_engine.foundation.data_types import (
        SystemConfig, TrainingConfig, ExperimentConfig, ExperimentResult
    )
    
    # Create in-memory tracker
    tracker = ResultTracker(db_path=':memory:')
    
    # Create a mock result
    system = SystemConfig(N=16, N_x=4, N_y=4, K=16, M=4)
    training = TrainingConfig(max_epochs=10)
    
    config = ExperimentConfig(
        name="test_experiment",
        system=system,
        training=training,
        probe_type="random_uniform",
        probe_params={},
        model_type="mlp",
        model_params={},
        data_source="synthetic_rayleigh",
        data_params={},
        metrics=["top_1_accuracy"]
    )
    
    result = ExperimentResult(
        config=config,
        metrics={"top_1_accuracy": 0.85},
        training_history={"train_loss": [1.0, 0.5], "val_loss": [1.0, 0.6]},
        best_epoch=1,
        total_epochs=2,
        training_time_seconds=10.5,
        model_parameters=1000,
        timestamp="2024-01-01T00:00:00",
        status="completed",
        primary_metric_name="top_1_accuracy",
        primary_metric_value=0.85
    )
    
    # Add result
    tracker.add_result(result)
    
    # Query results
    results = tracker.query(limit=10)
    assert len(results) == 1
    assert results[0].config.name == "test_experiment"
    
    print("✓ Result tracker working correctly")


def test_engine_components():
    """Test that engine components can be instantiated."""
    print("\nTesting engine components...")
    
    try:
        from ris_research_engine.engine.experiment_runner import ExperimentRunner
        from ris_research_engine.engine.search_controller import SearchController
        from ris_research_engine.engine.scientific_rules import RuleEngine
        from ris_research_engine.engine.result_analyzer import ResultAnalyzer
        from ris_research_engine.engine.report_generator import ReportGenerator
        from ris_research_engine.foundation.storage import ResultTracker
        
        tracker = ResultTracker(db_path=':memory:')
        
        # Instantiate components
        runner = ExperimentRunner(tracker)
        controller = SearchController(tracker)
        rules = RuleEngine()
        analyzer = ResultAnalyzer(tracker)
        reporter = ReportGenerator()
        
        print("✓ Engine components instantiated successfully")
    except (ImportError, OSError) as e:
        print(f"  ⚠ Engine test skipped (torch not available): {e.__class__.__name__}")


def test_ui_components():
    """Test UI component imports."""
    print("\nTesting UI components...")
    
    try:
        from ris_research_engine.ui.jupyter_minimal import RISEngine
        from ris_research_engine.ui import cli
        
        # Instantiate engine
        engine = RISEngine(db_path=':memory:')
        
        print("✓ UI components working correctly")
    except (ImportError, OSError) as e:
        print(f"  ⚠ UI test skipped (dependencies not available): {e.__class__.__name__}")


def test_configs():
    """Test that config files exist and are valid YAML."""
    print("\nTesting configuration files...")
    
    import yaml
    from pathlib import Path
    
    base_path = Path(__file__).parent.parent
    
    # Check presets
    preset_files = [
        'configs/presets/default_system.yaml',
        'configs/presets/default_training.yaml'
    ]
    
    for file_path in preset_files:
        full_path = base_path / file_path
        assert full_path.exists(), f"Missing config: {file_path}"
        
        with open(full_path) as f:
            config = yaml.safe_load(f)
            assert config is not None
    
    # Check search spaces
    search_files = [
        'configs/search_spaces/quick_test.yaml',
        'configs/search_spaces/probe_comparison.yaml'
    ]
    
    for file_path in search_files:
        full_path = base_path / file_path
        assert full_path.exists(), f"Missing config: {file_path}"
        
        with open(full_path) as f:
            config = yaml.safe_load(f)
            assert 'name' in config
            assert 'search_space' in config
    
    print("✓ Configuration files valid")


def main():
    """Run all tests."""
    print("="*60)
    print("RIS Auto-Research Engine - Smoke Tests")
    print("="*60)
    
    try:
        test_imports()
        test_plugin_discovery()
        test_data_types()
        test_result_tracker()
        test_engine_components()
        test_ui_components()
        test_configs()
        
        print("\n" + "="*60)
        print("✓ All smoke tests passed!")
        print("="*60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
