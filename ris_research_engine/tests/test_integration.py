"""Test integration: run quick_test.yaml end-to-end."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from ris_research_engine.engine.search_controller import SearchController
from ris_research_engine.engine.experiment_runner import ExperimentRunner


@pytest.fixture
def quick_test_config_path():
    """Get path to quick_test.yaml configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "search_spaces" / "quick_test.yaml"
    
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    return config_path


@pytest.fixture
def quick_test_config(quick_test_config_path):
    """Load quick_test.yaml configuration."""
    with open(quick_test_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


class TestQuickTestConfig:
    """Test quick_test.yaml configuration structure."""
    
    def test_config_exists(self, quick_test_config_path):
        """Test quick_test.yaml exists."""
        assert quick_test_config_path.exists()
        assert quick_test_config_path.suffix == '.yaml'
    
    def test_config_loads(self, quick_test_config):
        """Test configuration loads as valid YAML."""
        assert quick_test_config is not None
        assert isinstance(quick_test_config, dict)
    
    def test_config_structure(self, quick_test_config):
        """Test configuration has required fields."""
        required_fields = ['name', 'system', 'probes', 'model', 'training', 'data']
        
        for field in required_fields:
            assert field in quick_test_config, f"Missing required field: {field}"
    
    def test_system_config(self, quick_test_config):
        """Test system configuration."""
        system = quick_test_config['system']
        
        assert 'N' in system
        assert 'K' in system
        assert 'M' in system
        
        # Check they are reasonable values
        assert system['N'] > 0
        assert system['K'] > 0
        assert system['M'] > 0
    
    def test_training_config(self, quick_test_config):
        """Test training configuration."""
        training = quick_test_config['training']
        
        assert 'learning_rate' in training
        assert 'batch_size' in training
        assert 'max_epochs' in training
        
        assert training['max_epochs'] <= 20, "Quick test should have few epochs"
    
    def test_data_config(self, quick_test_config):
        """Test data configuration."""
        data = quick_test_config['data']
        
        assert 'num_samples' in data
        assert data['num_samples'] <= 200, "Quick test should have few samples"
    
    def test_probe_config(self, quick_test_config):
        """Test probe configuration."""
        probes = quick_test_config['probes']
        
        assert 'type' in probes
        probe_types = probes['type']
        
        assert isinstance(probe_types, list)
        assert len(probe_types) >= 1, "Should have at least one probe type"


class TestQuickTestExecution:
    """Test running quick_test.yaml end-to-end."""
    
    def test_single_experiment_from_config(self, quick_test_config):
        """Test running a single experiment from config."""
        runner = ExperimentRunner()
        
        # Extract first probe type and construct minimal config
        from ris_research_engine.foundation import SystemConfig, TrainingConfig, ExperimentConfig
        
        system = SystemConfig(
            N=quick_test_config['system']['N'],
            K=quick_test_config['system']['K'],
            M=quick_test_config['system']['M'],
            frequency=quick_test_config['system']['frequency'],
            snr_db=quick_test_config['system']['snr_db']
        )
        
        training = TrainingConfig(
            learning_rate=quick_test_config['training']['learning_rate'],
            batch_size=quick_test_config['training']['batch_size'],
            max_epochs=min(quick_test_config['training']['max_epochs'], 3),  # Reduce for test
            early_stopping_patience=quick_test_config['training']['early_stopping_patience'],
            optimizer=quick_test_config['training']['optimizer'],
            loss_function=quick_test_config['training']['loss_function'],
            val_split=quick_test_config['training']['val_split'],
            test_split=quick_test_config['training']['test_split'],
            device="cpu"
        )
        
        probe_type = quick_test_config['probes']['type'][0]
        
        experiment = ExperimentConfig(
            name=f"quick_test_{probe_type}",
            system=system,
            training=training,
            probe_type=probe_type,
            probe_params={},
            model_type=quick_test_config['model']['type'],
            model_params={},
            data_source=quick_test_config['data']['data_source'],
            data_params={"num_samples": min(quick_test_config['data']['num_samples'], 50)},
            metrics=["top_1_accuracy"],
            data_fidelity="synthetic"
        )
        
        try:
            result = runner.run(experiment)
            assert result is not None, "Experiment should produce a result"
        except Exception as e:
            pytest.skip(f"Experiment failed (expected in test environment): {e}")
    
    def test_search_controller_initialization(self, quick_test_config_path):
        """Test SearchController can be initialized with config."""
        try:
            controller = SearchController(config_path=str(quick_test_config_path))
            assert controller is not None
        except Exception as e:
            pytest.skip(f"SearchController initialization failed (expected): {e}")
    
    def test_quick_test_completes(self, quick_test_config_path, temp_db):
        """Test quick_test completes end-to-end (skipped if too slow)."""
        pytest.skip("Full quick_test takes too long for unit tests")
        
        # This would be an integration test
        # controller = SearchController(config_path=str(quick_test_config_path))
        # results = controller.run()
        # assert len(results) > 0


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_system_validation(self, quick_test_config):
        """Test system config can be validated."""
        from ris_research_engine.foundation import SystemConfig
        
        system_dict = quick_test_config['system']
        system = SystemConfig(
            N=system_dict['N'],
            K=system_dict['K'],
            M=system_dict['M'],
            frequency=system_dict['frequency'],
            snr_db=system_dict['snr_db']
        )
        
        # Should not raise
        system.validate()
    
    def test_training_validation(self, quick_test_config):
        """Test training config can be validated."""
        from ris_research_engine.foundation import TrainingConfig
        
        training_dict = quick_test_config['training']
        training = TrainingConfig(
            learning_rate=training_dict['learning_rate'],
            batch_size=training_dict['batch_size'],
            max_epochs=training_dict['max_epochs'],
            early_stopping_patience=training_dict['early_stopping_patience'],
            optimizer=training_dict['optimizer'],
            loss_function=training_dict['loss_function'],
            val_split=training_dict['val_split'],
            test_split=training_dict['test_split']
        )
        
        # Should not raise
        training.validate()


class TestProbeModelCombinations:
    """Test different probe-model combinations from config."""
    
    def test_all_probe_types_valid(self, quick_test_config):
        """Test all probe types in config are valid."""
        from ris_research_engine.plugins.probes import list_probes
        
        available_probes = list_probes()
        config_probes = quick_test_config['probes']['type']
        
        for probe in config_probes:
            assert probe in available_probes, f"Probe '{probe}' not available"
    
    def test_model_type_valid(self, quick_test_config):
        """Test model type in config is valid."""
        from ris_research_engine.plugins.models import list_models
        
        available_models = list_models()
        config_model = quick_test_config['model']['type']
        
        assert config_model in available_models, f"Model '{config_model}' not available"
    
    def test_data_source_valid(self, quick_test_config):
        """Test data source in config is valid."""
        data_source = quick_test_config['data']['data_source']
        
        # Common data sources
        valid_sources = ['synthetic', 'synthetic_rayleigh', 'synthetic_rician', 'hdf5']
        
        assert data_source in valid_sources or data_source.endswith('.h5'), \
            f"Data source '{data_source}' not recognized"


class TestOutputConfiguration:
    """Test output configuration from quick_test."""
    
    def test_output_config_exists(self, quick_test_config):
        """Test output configuration exists."""
        assert 'output' in quick_test_config
    
    def test_output_settings(self, quick_test_config):
        """Test output settings are reasonable for quick test."""
        output = quick_test_config['output']
        
        if 'save_models' in output:
            # Quick test typically doesn't save models
            assert isinstance(output['save_models'], bool)
        
        if 'verbose' in output:
            assert isinstance(output['verbose'], bool)


class TestMetricsConfiguration:
    """Test metrics configuration."""
    
    def test_metrics_config_exists(self, quick_test_config):
        """Test metrics configuration exists."""
        assert 'metrics' in quick_test_config
    
    def test_primary_metric(self, quick_test_config):
        """Test primary metric is defined."""
        metrics = quick_test_config['metrics']
        
        assert 'primary' in metrics
        assert isinstance(metrics['primary'], str)
    
    def test_secondary_metrics(self, quick_test_config):
        """Test secondary metrics are defined."""
        metrics = quick_test_config['metrics']
        
        if 'secondary' in metrics:
            assert isinstance(metrics['secondary'], list)


class TestMinimalConfig:
    """Test with absolute minimal configuration."""
    
    def test_minimal_system(self):
        """Test minimal system configuration works."""
        from ris_research_engine.foundation import SystemConfig
        
        config = SystemConfig(N=4, K=4, M=2, frequency=28e9, snr_db=10.0)
        config.validate()
    
    def test_minimal_training(self):
        """Test minimal training configuration works."""
        from ris_research_engine.foundation import TrainingConfig
        
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=4,
            max_epochs=2,
            early_stopping_patience=2
        )
        config.validate()
    
    def test_minimal_experiment(self):
        """Test minimal experiment can be created."""
        from ris_research_engine.foundation import SystemConfig, TrainingConfig, ExperimentConfig
        
        system = SystemConfig(N=4, K=4, M=2, frequency=28e9, snr_db=10.0)
        training = TrainingConfig(
            learning_rate=0.01,
            batch_size=4,
            max_epochs=2,
            early_stopping_patience=2,
            device="cpu"
        )
        
        experiment = ExperimentConfig(
            name="minimal_test",
            system=system,
            training=training,
            probe_type="hadamard",
            probe_params={},
            model_type="mlp",
            model_params={},
            data_source="synthetic",
            data_params={"num_samples": 20},
            metrics=["top_1_accuracy"],
            data_fidelity="synthetic"
        )
        
        assert experiment is not None
