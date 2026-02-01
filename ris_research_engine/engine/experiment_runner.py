"""Experiment runner for executing single RIS experiments."""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ris_research_engine.foundation import (
    SystemConfig, TrainingConfig, ExperimentConfig, ExperimentResult, ResultTracker
)
from ris_research_engine.plugins.probes import get_probe
from ris_research_engine.plugins.models import get_model
from ris_research_engine.plugins.data_sources import get_data_source
from ris_research_engine.plugins.metrics import get_metric
from ris_research_engine.plugins.baselines import AVAILABLE_BASELINES

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runner for executing single experiments."""
    
    def __init__(self, db_path: str = "ris_results.db"):
        """
        Initialize experiment runner.
        
        Args:
            db_path: Path to SQLite database for storing results
        """
        self.db_path = db_path
        self.tracker = ResultTracker(db_path)
    
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment with the given configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            ExperimentResult with metrics and training history
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            # Validate configurations
            config.system.validate()
            
            # Set device
            device = self._get_device(config.training.device)
            logger.info(f"Using device: {device}")
            
            # Set random seed
            self._set_seed(config.training.random_seed)
            
            # Load data
            logger.info(f"Loading data from {config.data_source}...")
            data = self._load_data(config)
            
            # Generate probes
            logger.info(f"Generating probes: {config.probe_type}...")
            probe_matrix = self._generate_probes(config)
            
            # Apply probe matrix to data
            train_inputs = self._apply_probes(data['train_inputs'], probe_matrix)
            val_inputs = self._apply_probes(data['val_inputs'], probe_matrix)
            test_inputs = self._apply_probes(data['test_inputs'], probe_matrix)
            
            # Build model
            logger.info(f"Building model: {config.model_type}...")
            output_dim = data['train_targets'].shape[1] if data['train_targets'].ndim > 1 else config.system.K
            model = self._build_model(config, train_inputs.shape[1], output_dim)
            model = model.to(device)
            
            # Count parameters
            model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model has {model_parameters:,} trainable parameters")
            
            # Create data loaders
            train_loader = self._create_dataloader(
                train_inputs, data['train_targets'], 
                config.training.batch_size, shuffle=True, 
                num_workers=config.training.num_workers
            )
            val_loader = self._create_dataloader(
                val_inputs, data['val_targets'],
                config.training.batch_size, shuffle=False,
                num_workers=config.training.num_workers
            )
            test_loader = self._create_dataloader(
                test_inputs, data['test_targets'],
                config.training.batch_size, shuffle=False,
                num_workers=config.training.num_workers
            )
            
            # Train model
            logger.info("Training model...")
            training_history, best_epoch = self._train_model(
                model, train_loader, val_loader, config.training, device
            )
            
            # Evaluate model
            logger.info("Evaluating model...")
            metrics = self._evaluate_model(
                model, test_loader, config.metrics, device,
                data['test_powers'], data.get('codebook')
            )
            
            # Run baselines
            logger.info("Running baseline comparisons...")
            baseline_results = self._run_baselines(
                data, probe_matrix, config.metrics
            )
            
            # Create result
            training_time = time.time() - start_time
            result = ExperimentResult(
                config=config,
                metrics=metrics,
                training_history=training_history,
                best_epoch=best_epoch,
                total_epochs=len(training_history.get('train_loss', [])),
                training_time_seconds=training_time,
                model_parameters=model_parameters,
                timestamp=timestamp,
                status='completed',
                error_message='',
                baseline_results=baseline_results,
                primary_metric_name=config.metrics[0] if config.metrics else 'top_1_accuracy',
                primary_metric_value=metrics.get(config.metrics[0] if config.metrics else 'top_1_accuracy', 0.0)
            )
            
            # Save to database
            self.tracker.save_experiment(result)
            logger.info(f"Experiment completed successfully in {training_time:.2f}s")
            
            return result
            
        except Exception as e:
            # Handle errors gracefully
            logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            training_time = time.time() - start_time
            
            # Create failed result
            result = ExperimentResult(
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
                primary_metric_name='',
                primary_metric_value=0.0
            )
            
            # Save failed result
            self.tracker.save_experiment(result)
            
            return result
    
    def _get_device(self, device_str: str) -> torch.device:
        """Get PyTorch device."""
        if device_str == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_str)
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _load_data(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Load data from data source plugin."""
        data_source = get_data_source(config.data_source)
        data_params = config.data_params.copy() if config.data_params else {}
        return data_source.load(config.system, **data_params)
    
    def _generate_probes(self, config: ExperimentConfig) -> np.ndarray:
        """Generate probe matrix using probe plugin."""
        probe = get_probe(config.probe_type)
        probe_params = config.probe_params.copy() if config.probe_params else {}
        return probe.generate(config.system.N, config.system.M, **probe_params)
    
    def _apply_probes(self, data: np.ndarray, probe_matrix: np.ndarray) -> np.ndarray:
        """Apply probe matrix to extract measurements from full data."""
        # Data is shape (n_samples, K) - powers from all K codebook entries
        # We need to extract M measurements
        # For simplicity, we just take the first M measurements
        # In practice, probe_matrix would be used to compute actual measurements
        return data[:, :probe_matrix.shape[0]]
    
    def _build_model(self, config: ExperimentConfig, input_dim: int, output_dim: int) -> nn.Module:
        """Build neural network model using model plugin."""
        model_class = get_model(config.model_type)
        model_params = config.model_params.copy() if config.model_params else {}
        model_params['dropout'] = config.training.dropout
        return model_class.build(input_dim, output_dim, **model_params)
    
    def _create_dataloader(
        self, inputs: np.ndarray, targets: np.ndarray,
        batch_size: int, shuffle: bool, num_workers: int
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        inputs_tensor = torch.FloatTensor(inputs)
        targets_tensor = torch.LongTensor(targets) if targets.ndim == 1 else torch.FloatTensor(targets)
        dataset = TensorDataset(inputs_tensor, targets_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    def _train_model(
        self, model: nn.Module, train_loader: DataLoader,
        val_loader: DataLoader, training_config: TrainingConfig, device: torch.device
    ) -> tuple:
        """Train the model with early stopping and learning rate scheduling."""
        # Setup optimizer
        if training_config.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
        elif training_config.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
        else:  # sgd
            optimizer = optim.SGD(model.parameters(), lr=training_config.learning_rate, 
                                 weight_decay=training_config.weight_decay, momentum=0.9)
        
        # Setup loss function
        if training_config.loss_function == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        elif training_config.loss_function == 'mse':
            criterion = nn.MSELoss()
        else:  # mae
            criterion = nn.L1Loss()
        
        # Setup learning rate scheduler
        if training_config.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config.max_epochs)
        elif training_config.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif training_config.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        else:
            scheduler = None
        
        # Training loop with early stopping
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(training_config.max_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Handle target shape for loss
                if training_config.loss_function == 'cross_entropy':
                    if targets.dim() > 1:
                        targets = targets.argmax(dim=1)
                
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                if training_config.gradient_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=training_config.gradient_clip_max_norm)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    
                    # Handle target shape for loss
                    targets_for_loss = targets
                    if training_config.loss_function == 'cross_entropy':
                        if targets.dim() > 1:
                            targets_for_loss = targets.argmax(dim=1)
                    
                    loss = criterion(outputs, targets_for_loss)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    if training_config.loss_function == 'cross_entropy':
                        _, predicted = outputs.max(1)
                        if targets.dim() > 1:
                            targets = targets.argmax(dim=1)
                        correct += predicted.eq(targets).sum().item()
                        total += targets.size(0)
            
            val_loss /= len(val_loader)
            val_accuracy = correct / total if total > 0 else 0.0
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Learning rate scheduling
            if scheduler is not None:
                if training_config.scheduler == 'plateau':
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
            
            if patience_counter >= training_config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{training_config.max_epochs}: "
                          f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                          f"val_acc={val_accuracy:.4f}")
        
        return history, best_epoch
    
    def _evaluate_model(
        self, model: nn.Module, test_loader: DataLoader,
        metric_names: list, device: torch.device,
        test_powers: Optional[np.ndarray] = None,
        codebook: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate model using specified metrics."""
        model.eval()
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Convert targets to indices if they're one-hot
        if all_targets.ndim > 1 and all_targets.shape[1] > 1:
            all_targets = all_targets.argmax(dim=1)
        
        # Calculate metrics
        results = {}
        for metric_name in metric_names:
            try:
                metric = get_metric(metric_name)
                # Metrics expect (predictions, targets, metadata=None)
                metadata = {
                    'test_powers': test_powers,
                    'codebook': codebook
                }
                value = metric.compute(all_outputs, all_targets, metadata)
                results[metric_name] = float(value)
            except Exception as e:
                logger.warning(f"Failed to compute metric {metric_name}: {e}")
                results[metric_name] = 0.0
        
        return results
    
    def _run_baselines(
        self, data: Dict[str, Any], probe_matrix: np.ndarray,
        metric_names: list
    ) -> Dict[str, Dict[str, float]]:
        """Run baseline algorithms for comparison."""
        baseline_results = {}
        
        # Extract test data
        test_powers = data.get('test_powers', data.get('test_inputs'))
        test_targets = data.get('test_targets')
        codebook = data.get('codebook')
        K = codebook.shape[0] if codebook is not None else test_powers.shape[1]
        
        # Apply probes to test data
        test_measurements = test_powers[:, :probe_matrix.shape[0]]
        
        for baseline_name, baseline_class in AVAILABLE_BASELINES.items():
            try:
                baseline = baseline_class()
                
                # Get baseline predictions for each sample
                # Baseline returns scores for all K configurations
                all_predictions = []
                for i in range(test_measurements.shape[0]):
                    scores = baseline.predict(
                        test_measurements[i],
                        np.arange(probe_matrix.shape[0]),
                        K
                    )
                    all_predictions.append(scores)
                
                predictions = np.array(all_predictions)
                
                # Convert to torch tensors for metric computation
                predictions_tensor = torch.FloatTensor(predictions)
                targets_tensor = torch.LongTensor(test_targets) if test_targets.ndim == 1 else torch.FloatTensor(test_targets)
                
                # Calculate metrics
                metrics = {}
                for metric_name in metric_names:
                    try:
                        metric = get_metric(metric_name)
                        metadata = {
                            'test_powers': test_powers,
                            'codebook': codebook
                        }
                        value = metric.compute(predictions_tensor, targets_tensor, metadata)
                        metrics[metric_name] = float(value)
                    except Exception as e:
                        logger.debug(f"Failed to compute {metric_name} for baseline {baseline_name}: {e}")
                        metrics[metric_name] = 0.0
                
                baseline_results[baseline_name] = metrics
            except Exception as e:
                logger.warning(f"Failed to run baseline {baseline_name}: {e}")
        
        return baseline_results
