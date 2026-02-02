"""Experiment runner for single experiment execution."""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from typing import Optional, Callable, Dict, List, Any
from pathlib import Path

from ris_research_engine.foundation import (
    SystemConfig, TrainingConfig, ExperimentConfig, ExperimentResult
)
from ris_research_engine.foundation.storage import ResultTracker
from ris_research_engine.foundation.logging_config import get_logger
from ris_research_engine.plugins.probes import get_probe
from ris_research_engine.plugins.models import get_model
from ris_research_engine.plugins.metrics import get_metric
from ris_research_engine.plugins.data_sources import get_data_source
from ris_research_engine.plugins.baselines import AVAILABLE_BASELINES

logger = get_logger(__name__)


def seed_everything(seed: int):
    """Seed all random number generators for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_baselines():
    """List all available baseline names."""
    return [name.lower() for name in AVAILABLE_BASELINES.keys()]


def get_baseline(name: str):
    """Get a baseline instance by name."""
    for key, cls in AVAILABLE_BASELINES.items():
        if key.lower() == name.lower():
            return cls()
    raise KeyError(f"Baseline '{name}' not found. Available: {list_baselines()}")


class ExperimentRunner:
    """Runs complete ML experiments for RIS configuration prediction."""
    
    def __init__(self, db_path: str = "outputs/experiments/results.db"):
        """Initialize the experiment runner.
        
        Args:
            db_path: Path to SQLite database for storing results
        """
        self.db_path = db_path
        self.tracker = ResultTracker(db_path)
        self.device = None
        self.current_model = None
    
    def _setup_device(self, device_config: str) -> torch.device:
        """Setup compute device."""
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        logger.info(f"Using device: {device}")
        return device
    
    def _create_dataloaders(self, data: Dict, batch_size: int, num_workers: int = 0):
        """Create PyTorch DataLoaders from data dict."""
        train_dataset = TensorDataset(
            torch.FloatTensor(data['train_inputs']),
            torch.LongTensor(data['train_targets'])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(data['val_inputs']),
            torch.LongTensor(data['val_targets'])
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(data['test_inputs']),
            torch.LongTensor(data['test_targets'])
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
    
    def _create_optimizer(self, model: nn.Module, config: TrainingConfig):
        """Create optimizer based on config."""
        if config.optimizer.lower() == 'adam':
            return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer.lower() == 'adamw':
            return optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer.lower() == 'sgd':
            return optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
        else:
            return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    def _create_scheduler(self, optimizer, config: TrainingConfig):
        """Create learning rate scheduler."""
        if config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
        elif config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        else:
            return None
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                     optimizer, criterion, device) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _validate(self, model: nn.Module, val_loader: DataLoader, 
                  criterion, device) -> tuple:
        """Validate model."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / max(len(val_loader), 1)
        accuracy = correct / max(total, 1)
        
        return avg_loss, accuracy
    
    def _evaluate_metrics(self, model: nn.Module, test_loader: DataLoader,
                          metric_names: List[str], device, data: Dict) -> Dict[str, float]:
        """Evaluate all specified metrics on test set."""
        model.eval()
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                all_outputs.append(outputs.cpu())
                all_targets.append(targets)
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = {}
        for metric_name in metric_names:
            try:
                metric = get_metric(metric_name)
                # Prepare metadata for metrics that need it
                metadata = {
                    'test_powers': data.get('test_powers', None),
                    'snr_db': data.get('metadata', {}).get('snr_db', 20.0)
                }
                value = metric.compute(all_outputs, all_targets, metadata)
                metrics[metric_name] = float(value)
            except Exception as e:
                logger.warning(f"Failed to compute metric {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        return metrics
    
    def _evaluate_baselines(self, data: Dict, system_config: SystemConfig) -> Dict[str, Dict[str, float]]:
        """Evaluate all baseline methods."""
        baseline_results = {}
        
        test_inputs = data['test_inputs']
        test_targets = data['test_targets']
        probe_indices = data.get('probe_indices', np.arange(system_config.M))
        
        for baseline_name in list_baselines():
            try:
                baseline = get_baseline(baseline_name)
                
                # Compute baseline predictions for each test sample
                correct_top1 = 0
                correct_top5 = 0
                total = len(test_targets)
                
                for i in range(total):
                    probe_measurements = test_inputs[i]
                    
                    # Get baseline prediction scores
                    scores = baseline.predict(
                        probe_measurements=np.array(probe_measurements),
                        probe_indices=np.array(probe_indices),
                        K=system_config.K
                    )
                    
                    # Get top predictions
                    top_indices = np.argsort(scores)[::-1]
                    true_label = int(test_targets[i])
                    
                    if top_indices[0] == true_label:
                        correct_top1 += 1
                    if true_label in top_indices[:5]:
                        correct_top5 += 1
                
                baseline_results[baseline_name] = {
                    'top_1_accuracy': correct_top1 / max(total, 1),
                    'top_5_accuracy': correct_top5 / max(total, 1)
                }
            except Exception as e:
                logger.warning(f"Failed to evaluate baseline {baseline_name}: {e}")
                baseline_results[baseline_name] = {'top_1_accuracy': 0.0, 'top_5_accuracy': 0.0}
        
        return baseline_results
    
    def run(self, config: ExperimentConfig, 
            progress_callback: Optional[Callable[[int, int, Dict], None]] = None) -> ExperimentResult:
        """Run a single experiment end-to-end.
        
        Args:
            config: Experiment configuration
            progress_callback: Optional callback(epoch, total_epochs, metrics_dict)
            
        Returns:
            ExperimentResult with all metrics and training history
        """
        timestamp = datetime.now().isoformat()
        start_time = time.time()
        
        try:
            # Seed for reproducibility
            seed_everything(config.training.random_seed)
            
            # Setup device
            self.device = self._setup_device(config.training.device)
            
            # Load data
            logger.info(f"Loading data from {config.data_source}")
            data_source = get_data_source(config.data_source)
            data = data_source.load(config.system, **config.data_params)
            
            # Update data_fidelity from data source
            config.data_fidelity = getattr(data_source, 'fidelity', 'synthetic')
            
            # Generate probes and apply to data
            logger.info(f"Generating {config.probe_type} probes")
            probe = get_probe(config.probe_type)
            probe_matrix = probe.generate(N=config.system.N, M=config.system.M, **config.probe_params)
            
            # Store probe indices for baseline evaluation
            data['probe_indices'] = np.arange(config.system.M)
            
            # Create data loaders
            train_loader, val_loader, test_loader = self._create_dataloaders(
                data, config.training.batch_size, config.training.num_workers
            )
            
            # Build model
            logger.info(f"Building {config.model_type} model")
            model_builder = get_model(config.model_type)
            # Get input dimension from actual data
            input_dim = data['train_inputs'].shape[1]
            output_dim = config.system.K  # K codebook configurations
            
            model = model_builder.build(input_dim, output_dim, **config.model_params)
            model = model.to(self.device)
            self.current_model = model
            
            # Count parameters
            model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model has {model_parameters:,} trainable parameters")
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = self._create_optimizer(model, config.training)
            scheduler = self._create_scheduler(optimizer, config.training)
            
            # Training loop
            training_history = {
                'train_loss': [],
                'val_loss': [],
                'val_accuracy': []
            }
            
            best_val_loss = float('inf')
            best_epoch = 0
            patience_counter = 0
            
            for epoch in range(config.training.max_epochs):
                # Train
                train_loss = self._train_epoch(model, train_loader, optimizer, criterion, self.device)
                
                # Validate
                val_loss, val_acc = self._validate(model, val_loader, criterion, self.device)
                
                # Record history
                training_history['train_loss'].append(train_loss)
                training_history['val_loss'].append(val_loss)
                training_history['val_accuracy'].append(val_acc)
                
                # Learning rate scheduling
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(epoch + 1, config.training.max_epochs, {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_accuracy': val_acc
                    })
                
                # Log progress
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"Epoch {epoch+1}/{config.training.max_epochs}: "
                               f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                
                # Early stopping
                if patience_counter >= config.training.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            total_epochs = len(training_history['train_loss'])
            
            # Evaluate metrics
            logger.info("Evaluating metrics on test set")
            metrics = self._evaluate_metrics(model, test_loader, config.metrics, self.device, data)
            
            # Evaluate baselines
            logger.info("Evaluating baseline methods")
            baseline_results = self._evaluate_baselines(data, config.system)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Determine primary metric
            primary_metric = config.metrics[0] if config.metrics else 'top_1_accuracy'
            primary_value = metrics.get(primary_metric, 0.0)
            
            # Create result
            result = ExperimentResult(
                config=config,
                metrics=metrics,
                training_history=training_history,
                best_epoch=best_epoch,
                total_epochs=total_epochs,
                training_time_seconds=training_time,
                model_parameters=model_parameters,
                timestamp=timestamp,
                status='completed',
                error_message='',
                baseline_results=baseline_results,
                primary_metric_name=primary_metric,
                primary_metric_value=primary_value
            )
            
            # Save to database
            exp_id = self.tracker.save_experiment(result)
            self.tracker.save_training_history(exp_id, training_history)
            logger.info(f"Experiment saved with ID: {exp_id}")
            
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            
            training_time = time.time() - start_time
            
            return ExperimentResult(
                config=config,
                metrics={},
                training_history={},
                best_epoch=0,
                total_epochs=0,
                training_time_seconds=training_time,
                model_parameters=0,
                timestamp=timestamp,
                status='failed',
                error_message=str(e),
                baseline_results={},
                primary_metric_name='top_1_accuracy',
                primary_metric_value=0.0
            )
