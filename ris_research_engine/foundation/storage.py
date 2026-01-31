"""Storage and persistence utilities for experiment results."""
import sqlite3
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
import threading

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

from .data_types import ExperimentResult, ExperimentConfig, SearchCampaignResult


class ResultTracker:
    """Thread-safe SQLite-based tracker for experiment results."""
    
    def __init__(self, db_path: str = "results.db"):
        """Initialize the result tracker.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.lock = threading.RLock()  # Use reentrant lock to allow nested calls
        # For in-memory databases, maintain a persistent connection
        self._conn = None
        if db_path == ':memory:':
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    timestamp TEXT,
                    status TEXT,
                    N INTEGER,
                    K INTEGER,
                    M INTEGER,
                    frequency REAL,
                    snr_db REAL,
                    probe_type TEXT,
                    model_type TEXT,
                    model_params TEXT,
                    data_source TEXT,
                    data_fidelity TEXT,
                    learning_rate REAL,
                    batch_size INTEGER,
                    max_epochs INTEGER,
                    best_epoch INTEGER,
                    total_epochs INTEGER,
                    training_time_seconds REAL,
                    model_parameters INTEGER,
                    metrics TEXT,
                    baseline_results TEXT,
                    campaign_name TEXT,
                    tags TEXT,
                    notes TEXT,
                    full_config TEXT
                )
            """)
            
            # Training history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    epoch INTEGER,
                    metric_name TEXT,
                    metric_value REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)
            
            # Campaigns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS campaigns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    timestamp TEXT,
                    search_strategy TEXT,
                    search_space TEXT,
                    total_experiments INTEGER,
                    completed INTEGER,
                    pruned INTEGER,
                    failed INTEGER,
                    best_experiment_id INTEGER,
                    total_time_seconds REAL,
                    FOREIGN KEY (best_experiment_id) REFERENCES experiments(id)
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a thread-safe database connection."""
        with self.lock:
            if self._conn is not None:
                # Use persistent connection for in-memory databases
                yield self._conn
            else:
                # Create new connection for file-based databases
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                try:
                    yield conn
                finally:
                    conn.close()
    
    def save_experiment(self, result: ExperimentResult, campaign_name: Optional[str] = None) -> int:
        """Save an experiment result to the database.
        
        Args:
            result: ExperimentResult object to save
            campaign_name: Optional campaign name to associate with this experiment
            
        Returns:
            experiment_id: The database ID of the saved experiment
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Extract configuration
            config = result.config
            system = config.system
            training = config.training
            
            # Insert experiment
            cursor.execute("""
                INSERT INTO experiments (
                    name, timestamp, status,
                    N, K, M, frequency, snr_db,
                    probe_type, model_type, model_params,
                    data_source, data_fidelity,
                    learning_rate, batch_size, max_epochs,
                    best_epoch, total_epochs, training_time_seconds,
                    model_parameters, metrics, baseline_results,
                    campaign_name, tags, notes, full_config
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config.name,
                result.timestamp,
                result.status,
                system.N,
                system.K,
                system.M,
                system.frequency,
                system.snr_db,
                config.probe_type,
                config.model_type,
                json.dumps(config.model_params),
                config.data_source,
                config.data_fidelity,
                training.learning_rate,
                training.batch_size,
                training.max_epochs,
                result.best_epoch,
                result.total_epochs,
                result.training_time_seconds,
                result.model_parameters,
                json.dumps(result.metrics),
                json.dumps(result.baseline_results),
                campaign_name,
                json.dumps(config.tags),
                config.notes,
                json.dumps(result.config.to_dict())
            ))
            
            experiment_id = cursor.lastrowid
            conn.commit()
            
            # Save training history
            self.save_training_history(experiment_id, result.training_history)
            
            return experiment_id
    
    def save_training_history(self, experiment_id: int, history: Dict[str, List[float]]):
        """Save training history for an experiment.
        
        Args:
            experiment_id: Database ID of the experiment
            history: Dictionary mapping metric names to lists of values per epoch
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare batch insert data
            records = []
            for metric_name, values in history.items():
                for epoch, value in enumerate(values):
                    records.append((experiment_id, epoch, metric_name, value))
            
            # Batch insert
            cursor.executemany("""
                INSERT INTO training_history (experiment_id, epoch, metric_name, metric_value)
                VALUES (?, ?, ?, ?)
            """, records)
            
            conn.commit()
    
    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve an experiment by ID.
        
        Args:
            experiment_id: Database ID of the experiment
            
        Returns:
            Dictionary containing experiment data, or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            # Convert to dictionary and parse JSON fields
            experiment = dict(row)
            experiment['model_params'] = json.loads(experiment['model_params'])
            experiment['metrics'] = json.loads(experiment['metrics'])
            experiment['baseline_results'] = json.loads(experiment['baseline_results'])
            experiment['tags'] = json.loads(experiment['tags'])
            experiment['full_config'] = json.loads(experiment['full_config'])
            
            # Get training history
            cursor.execute("""
                SELECT epoch, metric_name, metric_value
                FROM training_history
                WHERE experiment_id = ?
                ORDER BY epoch, metric_name
            """, (experiment_id,))
            
            history = {}
            for row in cursor.fetchall():
                metric_name = row['metric_name']
                if metric_name not in history:
                    history[metric_name] = []
                history[metric_name].append(row['metric_value'])
            
            experiment['training_history'] = history
            
            return experiment
    
    def get_all_experiments(self, 
                           campaign_name: Optional[str] = None,
                           status: Optional[str] = None,
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve all experiments, optionally filtered.
        
        Args:
            campaign_name: Filter by campaign name
            status: Filter by status ('completed', 'failed', 'pruned')
            limit: Maximum number of results to return
            
        Returns:
            List of experiment dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM experiments WHERE 1=1"
            params = []
            
            if campaign_name:
                query += " AND campaign_name = ?"
                params.append(campaign_name)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            
            experiments = []
            for row in cursor.fetchall():
                experiment = dict(row)
                experiment['model_params'] = json.loads(experiment['model_params'])
                experiment['metrics'] = json.loads(experiment['metrics'])
                experiment['baseline_results'] = json.loads(experiment['baseline_results'])
                experiment['tags'] = json.loads(experiment['tags'])
                experiment['full_config'] = json.loads(experiment['full_config'])
                experiments.append(experiment)
            
            return experiments
    
    def get_best_experiment(self, 
                           metric_name: str = 'top_1_accuracy',
                           campaign_name: Optional[str] = None,
                           maximize: bool = True) -> Optional[Dict[str, Any]]:
        """Get the best experiment based on a metric.
        
        Args:
            metric_name: Name of the metric to optimize
            campaign_name: Filter by campaign name
            maximize: If True, return experiment with highest metric; else lowest
            
        Returns:
            Best experiment dictionary, or None if no experiments found
        """
        experiments = self.get_all_experiments(campaign_name=campaign_name, status='completed')
        
        if not experiments:
            return None
        
        best_exp = None
        best_value = float('-inf') if maximize else float('inf')
        
        for exp in experiments:
            metrics = exp['metrics']
            if metric_name in metrics:
                value = metrics[metric_name]
                if maximize and value > best_value:
                    best_value = value
                    best_exp = exp
                elif not maximize and value < best_value:
                    best_value = value
                    best_exp = exp
        
        return best_exp
    
    def compare_experiments(self, experiment_ids: List[int]) -> Dict[str, Any]:
        """Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            Dictionary containing comparison data
        """
        experiments = []
        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                experiments.append(exp)
        
        if not experiments:
            return {'experiments': [], 'comparison': {}}
        
        # Collect all metric names
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp['metrics'].keys())
        
        # Build comparison table
        comparison = {
            'metric_names': sorted(all_metrics),
            'experiments': []
        }
        
        for exp in experiments:
            exp_data = {
                'id': exp['id'],
                'name': exp['name'],
                'timestamp': exp['timestamp'],
                'metrics': {m: exp['metrics'].get(m, None) for m in all_metrics}
            }
            comparison['experiments'].append(exp_data)
        
        return {
            'experiments': experiments,
            'comparison': comparison
        }
    
    def save_campaign(self, campaign: SearchCampaignResult) -> int:
        """Save a search campaign result.
        
        Args:
            campaign: SearchCampaignResult object to save
            
        Returns:
            campaign_id: The database ID of the saved campaign
        """
        # First, save all experiments and get best experiment ID
        best_experiment_id = None
        
        for result in campaign.all_results:
            exp_id = self.save_experiment(result, campaign_name=campaign.campaign_name)
            if campaign.best_result and result == campaign.best_result:
                best_experiment_id = exp_id
        
        # Save campaign metadata
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO campaigns (
                    name, timestamp, search_strategy, search_space,
                    total_experiments, completed, pruned, failed,
                    best_experiment_id, total_time_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                campaign.campaign_name,
                campaign.timestamp,
                campaign.search_strategy,
                json.dumps(campaign.search_space_definition),
                campaign.total_experiments,
                campaign.completed_experiments,
                campaign.pruned_experiments,
                campaign.failed_experiments,
                best_experiment_id,
                campaign.total_time_seconds
            ))
            
            campaign_id = cursor.lastrowid
            conn.commit()
            
            return campaign_id
    
    def get_campaign(self, campaign_id: Optional[int] = None, 
                     campaign_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a campaign by ID or name.
        
        Args:
            campaign_id: Database ID of the campaign
            campaign_name: Name of the campaign
            
        Returns:
            Dictionary containing campaign data, or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if campaign_id is not None:
                cursor.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,))
            elif campaign_name is not None:
                cursor.execute("SELECT * FROM campaigns WHERE name = ?", (campaign_name,))
            else:
                return None
            
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            campaign = dict(row)
            campaign['search_space'] = json.loads(campaign['search_space'])
            
            # Get associated experiments
            experiments = self.get_all_experiments(campaign_name=campaign['name'])
            campaign['experiments'] = experiments
            
            return campaign
    
    def export_to_csv(self, output_path: str, campaign_name: Optional[str] = None):
        """Export experiments to CSV file.
        
        Args:
            output_path: Path to output CSV file
            campaign_name: Optional campaign name filter
        """
        experiments = self.get_all_experiments(campaign_name=campaign_name)
        
        if not experiments:
            return
        
        # Collect all metric names
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp['metrics'].keys())
        
        metric_names = sorted(all_metrics)
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            fieldnames = [
                'id', 'name', 'timestamp', 'status',
                'N', 'K', 'M', 'frequency', 'snr_db',
                'probe_type', 'model_type', 'data_source', 'data_fidelity',
                'learning_rate', 'batch_size', 'max_epochs',
                'best_epoch', 'total_epochs', 'training_time_seconds',
                'model_parameters', 'campaign_name'
            ] + [f'metric_{m}' for m in metric_names]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for exp in experiments:
                row = {
                    'id': exp['id'],
                    'name': exp['name'],
                    'timestamp': exp['timestamp'],
                    'status': exp['status'],
                    'N': exp['N'],
                    'K': exp['K'],
                    'M': exp['M'],
                    'frequency': exp['frequency'],
                    'snr_db': exp['snr_db'],
                    'probe_type': exp['probe_type'],
                    'model_type': exp['model_type'],
                    'data_source': exp['data_source'],
                    'data_fidelity': exp['data_fidelity'],
                    'learning_rate': exp['learning_rate'],
                    'batch_size': exp['batch_size'],
                    'max_epochs': exp['max_epochs'],
                    'best_epoch': exp['best_epoch'],
                    'total_epochs': exp['total_epochs'],
                    'training_time_seconds': exp['training_time_seconds'],
                    'model_parameters': exp['model_parameters'],
                    'campaign_name': exp['campaign_name'],
                }
                
                # Add metrics
                for metric in metric_names:
                    row[f'metric_{metric}'] = exp['metrics'].get(metric, None)
                
                writer.writerow(row)
    
    def get_cross_fidelity_pairs(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Find experiments that can be compared across fidelities.
        
        Returns pairs of experiments (low_fidelity, high_fidelity) that have
        matching configurations but different data_fidelity levels.
        
        Returns:
            List of tuples (low_fidelity_exp, high_fidelity_exp)
        """
        experiments = self.get_all_experiments(status='completed')
        
        # Define fidelity hierarchy
        fidelity_order = {'synthetic': 0, 'sionna': 1, 'hardware': 2}
        
        # Group experiments by configuration (excluding data_fidelity)
        config_groups = {}
        
        for exp in experiments:
            # Create a key from configuration parameters
            config_key = (
                exp['N'], exp['K'], exp['M'],
                exp['frequency'], exp['snr_db'],
                exp['probe_type'], exp['model_type'],
                exp['data_source']
            )
            
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(exp)
        
        # Find pairs with different fidelities
        pairs = []
        
        for config_key, exps in config_groups.items():
            if len(exps) < 2:
                continue
            
            # Sort by fidelity
            exps_sorted = sorted(exps, key=lambda e: fidelity_order.get(e['data_fidelity'], -1))
            
            # Create pairs
            for i in range(len(exps_sorted) - 1):
                for j in range(i + 1, len(exps_sorted)):
                    low_fid = exps_sorted[i]
                    high_fid = exps_sorted[j]
                    
                    if low_fid['data_fidelity'] != high_fid['data_fidelity']:
                        pairs.append((low_fid, high_fid))
        
        return pairs


# HDF5 Helper Functions

def detect_hdf5_format(h5_path: str) -> str:
    """Auto-detect the format of an HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
        
    Returns:
        Format string: 'automl', 'session5', or 'generic'
    """
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 operations")
    
    with h5py.File(h5_path, 'r') as f:
        keys = set(f.keys())
        
        # Check for AutoML format
        if 'config' in keys and 'results' in keys and 'training_history' in keys:
            return 'automl'
        
        # Check for Session5 format
        if 'H' in keys and 'pilot_matrix' in keys and 'codewords' in keys:
            return 'session5'
        
        # Generic format
        return 'generic'


def load_hdf5_data(h5_path: str, format_hint: Optional[str] = None) -> Dict[str, Any]:
    """Load data from HDF5 file with auto-format detection.
    
    Args:
        h5_path: Path to HDF5 file
        format_hint: Optional format hint ('automl', 'session5', 'generic')
        
    Returns:
        Dictionary containing loaded data
    """
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 operations")
    
    # Auto-detect format if not provided
    if format_hint is None:
        format_hint = detect_hdf5_format(h5_path)
    
    data = {}
    
    with h5py.File(h5_path, 'r') as f:
        if format_hint == 'automl':
            # Load AutoML format
            if 'config' in f:
                config_str = f['config'][()].decode('utf-8')
                data['config'] = json.loads(config_str)
            
            if 'results' in f:
                results_str = f['results'][()].decode('utf-8')
                data['results'] = json.loads(results_str)
            
            if 'training_history' in f:
                history_group = f['training_history']
                data['training_history'] = {
                    key: history_group[key][:].tolist()
                    for key in history_group.keys()
                }
        
        elif format_hint == 'session5':
            # Load Session5 format
            for key in f.keys():
                dataset = f[key]
                if isinstance(dataset, h5py.Dataset):
                    data[key] = dataset[:]
                else:
                    # Nested group
                    data[key] = {
                        subkey: f[key][subkey][:]
                        for subkey in f[key].keys()
                    }
        
        else:
            # Generic format - load all datasets
            def load_group(group, prefix=''):
                for key in group.keys():
                    item = group[key]
                    full_key = f"{prefix}/{key}" if prefix else key
                    
                    if isinstance(item, h5py.Dataset):
                        data[full_key] = item[:]
                    elif isinstance(item, h5py.Group):
                        load_group(item, full_key)
            
            load_group(f)
    
    data['format'] = format_hint
    return data


def save_hdf5_data(h5_path: str, data: Dict[str, Any], format_type: str = 'automl'):
    """Save data to HDF5 file in specified format.
    
    Args:
        h5_path: Path to HDF5 file
        data: Dictionary containing data to save
        format_type: Format to use ('automl', 'generic')
    """
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 operations")
    
    with h5py.File(h5_path, 'w') as f:
        if format_type == 'automl':
            # Save in AutoML format
            if 'config' in data:
                config_str = json.dumps(data['config'])
                f.create_dataset('config', data=config_str.encode('utf-8'))
            
            if 'results' in data:
                results_str = json.dumps(data['results'])
                f.create_dataset('results', data=results_str.encode('utf-8'))
            
            if 'training_history' in data:
                history_group = f.create_group('training_history')
                for key, values in data['training_history'].items():
                    history_group.create_dataset(key, data=values)
        
        else:
            # Generic format - save all items
            for key, value in data.items():
                if isinstance(value, (list, tuple)):
                    import numpy as np
                    f.create_dataset(key, data=np.array(value))
                elif isinstance(value, dict):
                    # Create nested group
                    group = f.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (list, tuple)):
                            import numpy as np
                            group.create_dataset(subkey, data=np.array(subvalue))
                        else:
                            group.create_dataset(subkey, data=subvalue)
                else:
                    f.create_dataset(key, data=value)
