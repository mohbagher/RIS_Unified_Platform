"""Experiment runner for the RIS Auto-Research Engine."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import logging

from ris_research_engine.foundation import (
    SystemConfig, TrainingConfig, ExperimentConfig, 
    ExperimentResult, ResultTracker
)
from ris_research_engine.foundation.logging_config import get_logger
from ris_research_engine.plugins.probes import get_probe
from ris_research_engine.plugins.models import get_model
from ris_research_engine.plugins.metrics import get_metric
from ris_research_engine.plugins.data_sources import get_data_source
from ris_research_engine.plugins.baselines import AVAILABLE_BASELINES

logger = get_logger(__name__)


class ExperimentRunner:
    """Runs complete ML experiments for RIS configuration selection."""
    
    def __init__(self):
        """Initialize the experiment runner."""
        self.device = None
        self.current_experiment_id = None
    
    def run(
        self, 
        config: ExperimentConfig, 
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ExperimentResult:
        """
        Run a complete experiment with training, evaluation, and baseline comparison.
        
        Args:
            config: Experiment configuration
            progress_callback: Optional callback(message, progress) for UI updates
            
        Returns:
            ExperimentResult with metrics, training history, and baseline results
        """
        timestamp = datetime.now().isoformat()
        start_time = time.time()
        
        try:
            # Validate configurations
            config.system.validate()
            config.training.validate()
            
            # Setup device
            self.device = self._setup_device(config.training.device)
            
            # Progress update
            if progress_callback:
                progress_callback("Loading data...", 0.1)
            
            # Load data
            logger.info(f"Loading data from {config.data_source}")
            data_source = get_data_source(config.data_source)
            data = data_source.load(config.system, **config.data_params)
            
            # Progress update
            if progress_callback:
                progress_callback("Generating probes...", 0.2)
            
            # Generate probe configurations (codebook already in data)
            logger.info(f"Using probe type: {config.probe_type}")
            probe = get_probe(config.probe_type)
            
            # Progress update
            if progress_callback:
                progress_callback("Building model...", 0.3)
            
            # Build model
            logger.info(f"Building model: {config.model_type}")
            model_builder = get_model(config.model_type)
            input_dim = data['train_inputs'].shape[1]
            output_dim = config.system.K
            
            model_params = model_builder.get_default_params()
            model_params.update(config.model_params)
            model_params['dropout'] = config.training.dropout
            
            model = model_builder.build(input_dim, output_dim, **model_params)
            model = model.to(self.device)
            
            # Count parameters
            model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model has {model_parameters:,} trainable parameters")
            
            # Progress update
            if progress_callback:
                progress_callback("Training model...", 0.4)
            
            # Train model
            training_history, best_epoch = self._train_model(
                model, data, config.training, progress_callback
            )
            
            # Progress update
            if progress_callback:
                progress_callback("Evaluating metrics...", 0.8)
            
            # Evaluate metrics
            metrics = self._evaluate_metrics(
                model, data, config.metrics, config.system
            )
            
            # Progress update
            if progress_callback:
                progress_callback("Evaluating baselines...", 0.9)
            
            # Evaluate baselines
            baseline_results = self._evaluate_baselines(
                data, config.system
            )
            
            # Determine best epoch
            if 'val_acc' in training_history and len(training_history['val_acc']) > 0:
                best_epoch = int(np.argmax(training_history['val_acc']))
            else:
                best_epoch = len(training_history.get('train_loss', [])) - 1
            
            total_time = time.time() - start_time
            
            # Create result
            result = ExperimentResult(
                config=config,
                metrics=metrics,
                training_history=training_history,
                best_epoch=best_epoch,
                total_epochs=len(training_history.get('train_loss', [])),
                training_time_seconds=total_time,
                model_parameters=model_parameters,
                timestamp=timestamp,
                status='completed',
                error_message='',
                baseline_results=baseline_results,
                primary_metric_name='top_1_accuracy',
                primary_metric_value=metrics.get('top_1_accuracy', 0.0)
            )
            
            logger.info(f"Experiment completed successfully in {total_time:.2f}s")
            
            if progress_callback:
                progress_callback("Completed!", 1.0)
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            
            # Return failed result
            result = ExperimentResult(
                config=config,
                metrics={},
                training_history={},
                best_epoch=0,
                total_epochs=0,
                training_time_seconds=total_time,
                model_parameters=0,
                timestamp=timestamp,
                status='failed',
                error_message=str(e),
                baseline_results={},
                primary_metric_name='top_1_accuracy',
                primary_metric_value=0.0
            )
            
            if progress_callback:
                progress_callback(f"Failed: {str(e)}", 1.0)
            
            return result
    
    def _setup_device(self, device_config: str) -> torch.device:
        """Setup computation device."""
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _train_model(
        self, 
        model: nn.Module, 
        data: Dict[str, Any], 
        training_config: TrainingConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> tuple:
        """
        Train the model with early stopping and LR scheduling.
        
        Returns:
            Tuple of (training_history, best_epoch)
        """
        # Prepare data loaders
        train_loader = self._create_dataloader(
            data['train_inputs'], data['train_targets'],
            training_config.batch_size, shuffle=True,
            num_workers=training_config.num_workers
        )
        
        val_loader = self._create_dataloader(
            data['val_inputs'], data['val_targets'],
            training_config.batch_size, shuffle=False,
            num_workers=training_config.num_workers
        )
        
        # Setup optimizer
        if training_config.optimizer == 'adam':
            optimizer = optim.Adam(
                model.parameters(), 
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay
            )
        elif training_config.optimizer == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay
            )
        elif training_config.optimizer == 'sgd':
            optimizer = optim.SGD(
                model.parameters(), 
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {training_config.optimizer}")
        
        # Setup scheduler
        scheduler = None
        if training_config.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=training_config.max_epochs
            )
        elif training_config.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif training_config.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        logger.info(f"Starting training for {training_config.max_epochs} epochs")
        
        for epoch in range(training_config.max_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * batch_inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == batch_targets).sum().item()
                train_total += batch_targets.size(0)
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_inputs, batch_targets in val_loader:
                    batch_inputs = batch_inputs.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = model(batch_inputs)
                    loss = criterion(outputs, batch_targets)
                    
                    val_loss += loss.item() * batch_inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == batch_targets).sum().item()
                    val_total += batch_targets.size(0)
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Update scheduler
            if scheduler is not None:
                if training_config.scheduler == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{training_config.max_epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )
            
            # Progress callback
            if progress_callback:
                progress = 0.4 + 0.4 * (epoch + 1) / training_config.max_epochs
                progress_callback(
                    f"Epoch {epoch+1}/{training_config.max_epochs}", 
                    progress
                )
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= training_config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        best_epoch = int(np.argmin(history['val_loss']))
        
        return history, best_epoch
    
    def _create_dataloader(
        self, 
        inputs: np.ndarray, 
        targets: np.ndarray,
        batch_size: int,
        shuffle: bool,
        num_workers: int
    ) -> DataLoader:
        """Create a PyTorch DataLoader from numpy arrays."""
        inputs_tensor = torch.FloatTensor(inputs)
        targets_tensor = torch.LongTensor(targets)
        dataset = TensorDataset(inputs_tensor, targets_tensor)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
    def _evaluate_metrics(
        self, 
        model: nn.Module, 
        data: Dict[str, Any],
        metric_names: list,
        system_config: SystemConfig
    ) -> Dict[str, float]:
        """Evaluate all requested metrics on test set."""
        model.eval()
        
        # Get test predictions
        test_inputs = torch.FloatTensor(data['test_inputs']).to(self.device)
        
        with torch.no_grad():
            logits = model(test_inputs)
            predictions = logits.cpu().numpy()
        
        test_targets = data['test_targets']
        test_powers = data['test_powers']
        
        # Evaluate each metric
        metrics_dict = {}
        
        for metric_name in metric_names:
            try:
                metric = get_metric(metric_name)
                
                # Measure inference time if it's the inference_time metric
                if metric_name == 'inference_time':
                    start_time = time.time()
                    with torch.no_grad():
                        _ = model(test_inputs)
                    inference_time = (time.time() - start_time) / len(test_inputs) * 1000
                    metrics_dict[metric_name] = inference_time
                else:
                    value = metric.compute(
                        predictions=predictions,
                        targets=test_targets,
                        all_powers=test_powers
                    )
                    metrics_dict[metric_name] = float(value)
                
                logger.info(f"Metric {metric_name}: {metrics_dict[metric_name]:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to compute metric {metric_name}: {e}")
                metrics_dict[metric_name] = 0.0
        
        return metrics_dict
    
    def _evaluate_baselines(
        self, 
        data: Dict[str, Any],
        system_config: SystemConfig
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate baseline methods on the same test data."""
        baseline_results = {}
        
        # Get test data
        test_inputs = data['test_inputs']  # Shape: (N_test, M*N)
        test_targets = data['test_targets']
        test_powers = data['test_powers']
        
        N_test = len(test_targets)
        M = system_config.M
        N = system_config.N
        K = system_config.K
        
        # Extract probe measurements and indices from test inputs
        # Assuming inputs are flattened probe measurements: (M, N) -> (M*N,)
        # We need to reconstruct probe measurements for baselines
        
        # For simplicity, use the test_powers directly as "measurements"
        # Get the M probed configurations (first M indices)
        probe_indices = np.arange(M)
        
        baseline_names = ['RandomSelection', 'BestOfProbed', 'ExhaustiveSearch', 'StrongestBeam']
        
        for baseline_name in baseline_names:
            if baseline_name not in AVAILABLE_BASELINES:
                logger.warning(f"Baseline {baseline_name} not available")
                continue
            
            try:
                baseline_class = AVAILABLE_BASELINES[baseline_name]
                baseline = baseline_class()
                
                # Collect predictions for all test samples
                all_predictions = []
                
                for i in range(N_test):
                    # Get probe measurements for this sample
                    sample_powers = test_powers[i]  # Shape: (K,)
                    probe_measurements = sample_powers[probe_indices]  # Shape: (M,)
                    
                    # Get baseline predictions
                    scores = baseline.predict(probe_measurements, probe_indices, K)
                    all_predictions.append(scores)
                
                all_predictions = np.array(all_predictions)  # Shape: (N_test, K)
                
                # Compute top-1, top-5, top-10 accuracy
                top_1_acc = self._compute_topk_accuracy(all_predictions, test_targets, k=1)
                top_5_acc = self._compute_topk_accuracy(all_predictions, test_targets, k=5)
                top_10_acc = self._compute_topk_accuracy(all_predictions, test_targets, k=10)
                
                baseline_results[baseline_name.lower()] = {
                    'top_1_accuracy': float(top_1_acc),
                    'top_5_accuracy': float(top_5_acc),
                    'top_10_accuracy': float(top_10_acc)
                }
                
                logger.info(
                    f"Baseline {baseline_name}: "
                    f"top-1={top_1_acc:.4f}, top-5={top_5_acc:.4f}, top-10={top_10_acc:.4f}"
                )
                
            except Exception as e:
                logger.warning(f"Failed to evaluate baseline {baseline_name}: {e}")
                baseline_results[baseline_name.lower()] = {
                    'top_1_accuracy': 0.0,
                    'top_5_accuracy': 0.0,
                    'top_10_accuracy': 0.0
                }
        
        return baseline_results
    
    def _compute_topk_accuracy(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray, 
        k: int
    ) -> float:
        """Compute top-k accuracy."""
        top_k_indices = np.argsort(predictions, axis=1)[:, -k:]
        correct = 0
        for i, target in enumerate(targets):
            if target in top_k_indices[i]:
                correct += 1
        return correct / len(targets)
