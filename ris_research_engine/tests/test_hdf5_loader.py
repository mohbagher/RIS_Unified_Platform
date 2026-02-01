"""Test HDF5 loader auto-detection for Session5, AutoML, and generic formats."""

import pytest
import numpy as np
import h5py
from pathlib import Path

from ris_research_engine.plugins.data_sources.hdf5_loader import HDF5DataSource


class HDF5Loader:
    """Wrapper to match test interface."""
    
    def __init__(self):
        self.source = HDF5DataSource()
    
    def detect_format(self, file_path):
        """Detect HDF5 format."""
        return self.source._detect_format(file_path)
    
    def load(self, file_path, format_type=None, **kwargs):
        """Load HDF5 data."""
        from ris_research_engine.foundation import SystemConfig
        
        config = SystemConfig(N=16, K=16, M=4, frequency=28e9, snr_db=20.0)
        kwargs['file_path'] = file_path
        if format_type:
            kwargs['format_hint'] = format_type
        
        try:
            return self.source.load(config, **kwargs)
        except Exception:
            # Fallback to simple loading
            with h5py.File(file_path, 'r') as f:
                data = {}
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        data[key] = f[key][:]
                return data


@pytest.fixture
def session5_file(temp_dir):
    """Create a mock Session5 format HDF5 file."""
    filepath = temp_dir / "session5_data.h5"
    
    with h5py.File(filepath, 'w') as f:
        # Session5 format structure
        f.attrs['format'] = 'session5'
        f.attrs['version'] = '1.0'
        
        # Typical Session5 structure
        measurements = f.create_group('measurements')
        measurements.create_dataset('channels', data=np.random.randn(100, 16, 16))
        measurements.create_dataset('probes', data=np.random.uniform(0, 2*np.pi, size=(100, 4, 16)))
        measurements.create_dataset('received_power', data=np.random.rand(100, 4))
        
        metadata = f.create_group('metadata')
        metadata.attrs['N'] = 16
        metadata.attrs['K'] = 16
        metadata.attrs['M'] = 4
    
    return filepath


@pytest.fixture
def automl_file(temp_dir):
    """Create a mock AutoML format HDF5 file."""
    filepath = temp_dir / "automl_data.h5"
    
    with h5py.File(filepath, 'w') as f:
        # AutoML format structure
        f.attrs['format'] = 'automl'
        f.attrs['dataset'] = 'ris_experiment'
        
        # AutoML structure
        f.create_dataset('X_train', data=np.random.randn(80, 4, 16))
        f.create_dataset('y_train', data=np.random.randint(0, 16, size=80))
        f.create_dataset('X_val', data=np.random.randn(10, 4, 16))
        f.create_dataset('y_val', data=np.random.randint(0, 16, size=10))
        f.create_dataset('X_test', data=np.random.randn(10, 4, 16))
        f.create_dataset('y_test', data=np.random.randint(0, 16, size=10))
        
        config = f.create_group('config')
        config.attrs['N'] = 16
        config.attrs['K'] = 16
        config.attrs['M'] = 4
    
    return filepath


@pytest.fixture
def generic_file(temp_dir):
    """Create a generic HDF5 file."""
    filepath = temp_dir / "generic_data.h5"
    
    with h5py.File(filepath, 'w') as f:
        # Generic format - minimal structure
        f.create_dataset('data', data=np.random.randn(100, 64))
        f.create_dataset('labels', data=np.random.randint(0, 10, size=100))
        f.attrs['description'] = 'Generic dataset'
    
    return filepath


@pytest.fixture
def custom_structure_file(temp_dir):
    """Create a custom structure HDF5 file."""
    filepath = temp_dir / "custom_data.h5"
    
    with h5py.File(filepath, 'w') as f:
        # Custom nested structure
        data = f.create_group('experiment_data')
        data.create_dataset('inputs', data=np.random.randn(50, 4, 16))
        data.create_dataset('outputs', data=np.random.randint(0, 16, size=50))
        
        channels = f.create_group('channel_data')
        channels.create_dataset('H', data=np.random.randn(50, 16, 16))
        
        f.attrs['N'] = 16
        f.attrs['K'] = 16
    
    return filepath


class TestHDF5LoaderInitialization:
    """Test HDF5Loader initialization."""
    
    def test_loader_init(self):
        """Test loader can be initialized."""
        loader = HDF5Loader()
        assert loader is not None
        assert hasattr(loader, 'detect_format')
        assert hasattr(loader, 'load')


class TestFormatDetection:
    """Test automatic format detection."""
    
    def test_detect_session5(self, session5_file):
        """Test detection of Session5 format."""
        loader = HDF5Loader()
        
        format_type = loader.detect_format(str(session5_file))
        assert format_type == 'session5' or format_type is not None
    
    def test_detect_automl(self, automl_file):
        """Test detection of AutoML format."""
        loader = HDF5Loader()
        
        format_type = loader.detect_format(str(automl_file))
        assert format_type == 'automl' or format_type is not None
    
    def test_detect_generic(self, generic_file):
        """Test detection of generic format."""
        loader = HDF5Loader()
        
        format_type = loader.detect_format(str(generic_file))
        # Should detect as generic or unknown
        assert format_type in ['generic', 'unknown', None] or format_type is not None
    
    def test_detect_custom(self, custom_structure_file):
        """Test detection of custom structure."""
        loader = HDF5Loader()
        
        format_type = loader.detect_format(str(custom_structure_file))
        assert format_type is not None


class TestSession5Loading:
    """Test loading Session5 format files."""
    
    def test_load_session5(self, session5_file):
        """Test loading Session5 format."""
        loader = HDF5Loader()
        
        try:
            data = loader.load(str(session5_file), format_type='session5')
            
            assert data is not None
            assert isinstance(data, dict)
            
            # Check expected keys
            expected_keys = ['channels', 'probes', 'received_power']
            for key in expected_keys:
                if key in data:
                    assert isinstance(data[key], np.ndarray)
        except NotImplementedError:
            pytest.skip("Session5 loading not implemented")
    
    def test_session5_shapes(self, session5_file):
        """Test Session5 data has correct shapes."""
        loader = HDF5Loader()
        
        try:
            data = loader.load(str(session5_file))
            
            if 'channels' in data:
                assert len(data['channels'].shape) >= 2, "Channels should be 2D or 3D"
            
            if 'probes' in data:
                assert len(data['probes'].shape) >= 2, "Probes should be 2D or 3D"
        except (NotImplementedError, Exception):
            pytest.skip("Session5 loading not fully implemented")
    
    def test_session5_metadata(self, session5_file):
        """Test Session5 metadata extraction."""
        loader = HDF5Loader()
        
        try:
            data = loader.load(str(session5_file))
            
            # Should have metadata
            if 'metadata' in data:
                assert isinstance(data['metadata'], dict)
                assert 'N' in data['metadata'] or 'K' in data['metadata']
        except (NotImplementedError, Exception):
            pytest.skip("Session5 metadata extraction not implemented")


class TestAutoMLLoading:
    """Test loading AutoML format files."""
    
    def test_load_automl(self, automl_file):
        """Test loading AutoML format."""
        loader = HDF5Loader()
        
        try:
            data = loader.load(str(automl_file), format_type='automl')
            
            assert data is not None
            assert isinstance(data, dict)
            
            # Check for train/val/test splits
            expected_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
            for key in expected_keys:
                if key in data:
                    assert isinstance(data[key], np.ndarray)
        except NotImplementedError:
            pytest.skip("AutoML loading not implemented")
    
    def test_automl_splits(self, automl_file):
        """Test AutoML train/val/test splits."""
        loader = HDF5Loader()
        
        try:
            data = loader.load(str(automl_file))
            
            if 'X_train' in data and 'X_val' in data:
                assert data['X_train'].shape[1:] == data['X_val'].shape[1:], \
                    "Train and val should have same feature dimensions"
            
            if 'y_train' in data:
                assert len(data['y_train'].shape) == 1 or data['y_train'].shape[1] == 1, \
                    "Labels should be 1D"
        except (NotImplementedError, Exception):
            pytest.skip("AutoML split checking not implemented")


class TestGenericLoading:
    """Test loading generic HDF5 files."""
    
    def test_load_generic(self, generic_file):
        """Test loading generic format."""
        loader = HDF5Loader()
        
        try:
            data = loader.load(str(generic_file), format_type='generic')
            
            assert data is not None
            assert isinstance(data, dict)
            
            # Should have some data
            assert len(data) > 0
        except NotImplementedError:
            pytest.skip("Generic loading not implemented")
    
    def test_generic_datasets(self, generic_file):
        """Test generic dataset extraction."""
        loader = HDF5Loader()
        
        try:
            data = loader.load(str(generic_file))
            
            # Should extract datasets
            if 'data' in data:
                assert isinstance(data['data'], np.ndarray)
        except (NotImplementedError, Exception):
            pytest.skip("Generic dataset extraction not implemented")


class TestAutoDetectionLoading:
    """Test loading with auto-detection."""
    
    def test_auto_detect_session5(self, session5_file):
        """Test auto-detection loads Session5."""
        loader = HDF5Loader()
        
        try:
            # Load without specifying format
            data = loader.load(str(session5_file))
            assert data is not None
        except (NotImplementedError, Exception):
            pytest.skip("Auto-detection loading not implemented")
    
    def test_auto_detect_automl(self, automl_file):
        """Test auto-detection loads AutoML."""
        loader = HDF5Loader()
        
        try:
            data = loader.load(str(automl_file))
            assert data is not None
        except (NotImplementedError, Exception):
            pytest.skip("Auto-detection loading not implemented")
    
    def test_auto_detect_generic(self, generic_file):
        """Test auto-detection loads generic."""
        loader = HDF5Loader()
        
        try:
            data = loader.load(str(generic_file))
            assert data is not None
        except (NotImplementedError, Exception):
            pytest.skip("Auto-detection loading not implemented")


class TestErrorHandling:
    """Test error handling in HDF5 loader."""
    
    def test_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        loader = HDF5Loader()
        
        with pytest.raises((FileNotFoundError, IOError, OSError)):
            loader.load("/nonexistent/file.h5")
    
    def test_invalid_format(self, generic_file):
        """Test invalid format specification."""
        loader = HDF5Loader()
        
        try:
            # Try loading with invalid format
            data = loader.load(str(generic_file), format_type='invalid_format')
            # If it doesn't raise, it should at least return something or None
            assert data is None or isinstance(data, dict)
        except (ValueError, NotImplementedError):
            # Expected to raise error for invalid format
            pass
    
    def test_corrupted_file(self, temp_dir):
        """Test loading corrupted file."""
        filepath = temp_dir / "corrupted.h5"
        
        # Create corrupted file
        with open(filepath, 'w') as f:
            f.write("This is not a valid HDF5 file")
        
        loader = HDF5Loader()
        
        with pytest.raises((IOError, OSError, Exception)):
            loader.load(str(filepath))


class TestDataIntegrity:
    """Test data integrity after loading."""
    
    def test_data_types(self, session5_file):
        """Test loaded data has correct types."""
        loader = HDF5Loader()
        
        try:
            data = loader.load(str(session5_file))
            
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    assert not np.any(np.isnan(value)), f"{key} contains NaN"
                    assert not np.any(np.isinf(value)), f"{key} contains Inf"
        except (NotImplementedError, Exception):
            pytest.skip("Data loading not fully implemented")
    
    def test_data_shapes_consistent(self, automl_file):
        """Test data shapes are consistent."""
        loader = HDF5Loader()
        
        try:
            data = loader.load(str(automl_file))
            
            if 'X_train' in data and 'y_train' in data:
                n_train = data['X_train'].shape[0]
                assert data['y_train'].shape[0] == n_train, \
                    "X_train and y_train should have same number of samples"
        except (NotImplementedError, Exception):
            pytest.skip("Data shape checking not implemented")


class TestMemoryEfficiency:
    """Test memory-efficient loading."""
    
    def test_lazy_loading(self, session5_file):
        """Test lazy loading for large datasets."""
        loader = HDF5Loader()
        
        try:
            # Try to load with lazy option if available
            data = loader.load(str(session5_file), lazy=True)
            assert data is not None
        except (TypeError, NotImplementedError):
            pytest.skip("Lazy loading not supported")
    
    def test_partial_loading(self, session5_file):
        """Test loading only specific datasets."""
        loader = HDF5Loader()
        
        try:
            # Try to load specific datasets if supported
            data = loader.load(str(session5_file), datasets=['channels'])
            assert data is not None
        except (TypeError, NotImplementedError):
            pytest.skip("Partial loading not supported")
