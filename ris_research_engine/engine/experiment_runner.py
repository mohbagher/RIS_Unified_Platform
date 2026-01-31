"""Experiment runner for conducting single RIS experiments."""

import time
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ris_research_engine.foundation.data_types import (
    ExperimentConfig, ExperimentResult
)
from ris_research_engine.foundation.storage import ResultTracker
from ris_research_engine.plugins.probes import get_probe
from ris_research_engine.plugins.models import get_model
from ris_research_engine.plugins.metrics import get_metric
from ris_research_engine.plugins.data_sources import get_data_source
from ris_research_engine.plugins.baselines import AVAILABLE_BASELINES

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run a single RIS experiment with full training and evaluation."""
    
    def __init__(self, result_tracker: Optional[ResultTracker] = None):
        """Initialize experiment runner.
        
        Args:
            result_tracker: Optional result tracker for saving results
        """
        self.result_tracker = result_tracker
        self.device = None
    
    def _setup_device(self, config: ExperimentConfig) -> torch.device:
        """Setup compute device based on config."""
        if config.training.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(config.training.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _load_data(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Load or generate data using data_source plugin."""
        logger.info(f"Loading data from {config.data_source}")
        
        data_source = get_data_source(config.data_source)
        data = data_source.load(config.system, **config.data_params)
        
        return data
    
    def _generate_probes(self, config: ExperimentConfig) -> np.ndarray:
        """Generate probe matrix using probe plugin."""
        logger.info(f"Generating probes: {config.probe_type}")
        
        probe = get_probe(config.probe_type)
        probe_matrix = probe.generate(
            config.system.N, 
            config.system.M, 
            **config.probe_params
        )
        
        return probe_matrix
    
    def _apply_probes(
        self, 
        data: Dict[str, Any], 
        probe_matrix: np.ndarray,
        config: ExperimentConfig
    ) -> Dict[str, np.ndarray]:
        """Apply probes to select M measurements from each sample.
        
        This simulates the RIS measurement process where we can only
        measure power for M specific phase configurations (probes),
        not all K codebook entries.
        """
        logger.info(f"Applying {config.system.M} probes to data")
        
        # For training, validation, and test sets, we extract the M probe measurements
        # In reality: probe_matrix (M x N) defines M phase configurations
        # We measure power for these M configurations from the full channel
        # Here we simulate by selecting the corresponding measurements
        
        # The data contains full power measurements for all K codebook entries
        # We select the M entries that correspond to our probes
        # This is a simplification - in real scenario, probes != codebook entries
        
        # For synthetic data, we sample M measurements from K available
        result = {}
        for split in ['train', 'val', 'test']:
            powers_key = f'{split}_powers'
            if powers_key in data:
                # Sample M measurements per sample
                # In practice, this would be the actual probe measurements
                powers = data[powers_key]  # (N_samples, K)
                N_samples, K = powers.shape
                
                # Randomly select M indices for each sample to simulate probe measurements
                # In a real scenario, this would come from the probe matrix
                np.random.seed(config.training.random_seed)
                probe_indices = np.random.choice(K, config.system.M, replace=False)
                probed_powers = powers[:, probe_indices]  # (N_samples, M)
                
                result[f'{split}_inputs'] = probed_powers
                result[f'{split}_targets'] = data[f'{split}_targets']
                result[f'{split}_powers'] = data[f'{split}_powers']
        
        return result
    
    def _build_model(self, config: ExperimentConfig) -> nn.Module:
        """Build model using model plugin."""
        logger.info(f"Building model: {config.model_type}")
        
        model_builder = get_model(config.model_type)
        model = model_builder.build(
            input_dim=config.system.M,
            output_dim=config.system.K,
            **config.model_params
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,}")
        
        return model
    
    def _setup_training(
        self, 
        model: nn.Module, 
        config: ExperimentConfig
    ) -> Tuple[optim.Optimizer, Any, nn.Module]:
        """Setup optimizer, scheduler, and loss function."""
        # Optimizer
        if config.training.optimizer == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )
        elif config.training.optimizer == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )
        elif config.training.optimizer == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=config.training.learning_rate,
                momentum=0.9,
                weight_decay=config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.training.optimizer}")
        
        # Scheduler
        if config.training.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=config.training.max_epochs
            )
        elif config.training.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=30, 
                gamma=0.1
            )
        elif config.training.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max', 
                patience=10, 
                factor=0.5
            )
        else:
            scheduler = None
        
        # Loss function
        if config.training.loss_function == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        elif config.training.loss_function == 'mse':
            criterion = nn.MSELoss()
        elif config.training.loss_function == 'mae':
            criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {config.training.loss_function}")
        
        return optimizer, scheduler, criterion
    
    def _create_dataloaders(
        self, 
        data: Dict[str, np.ndarray], 
        config: ExperimentConfig
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch dataloaders."""
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
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        
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
        
        return total_loss / len(train_loader)
    
    def _evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        metrics: list
    ) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                all_preds.append(outputs.cpu())
                all_targets.append(targets)
        
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute all metrics
        results = {}
        for metric_name in metrics:
            metric = get_metric(metric_name)
            results[metric_name] = metric.compute(preds, targets)
        
        return results
    
    def _train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Any,
        criterion: nn.Module,
        config: ExperimentConfig,
        device: torch.device,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Dict[str, list], int, int]:
        """Train model with early stopping."""
        history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        # Add metric tracking
        for metric_name in config.metrics:
            history[f'val_{metric_name}'] = []
        
        best_val_metric = -float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(config.training.max_epochs):
            # Train
            train_loss = self._train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self._evaluate(model, val_loader, device, config.metrics)
            
            # Compute validation loss
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            # Track metrics
            for metric_name, value in val_metrics.items():
                history[f'val_{metric_name}'].append(value)
            
            # Use primary metric for early stopping (default: top_1_accuracy)
            primary_metric = 'top_1_accuracy' if 'top_1_accuracy' in config.metrics else config.metrics[0]
            current_val_metric = val_metrics[primary_metric]
            
            # Check for improvement
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_val_metric)
                else:
                    scheduler.step()
            
            # Progress callback
            if progress_callback:
                progress_callback({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'best_epoch': best_epoch + 1,
                })
            
            # Log
            logger.info(
                f"Epoch {epoch+1}/{config.training.max_epochs} - "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                f"{primary_metric}: {current_val_metric:.4f}"
            )
            
            # Early stopping
            if patience_counter >= config.training.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        total_epochs = epoch + 1
        return history, best_epoch, total_epochs
    
    def _evaluate_baselines(
        self,
        data: Dict[str, np.ndarray],
        config: ExperimentConfig
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all baseline methods on test set."""
        logger.info("Evaluating baseline methods")
        
        baseline_results = {}
        
        # Get test data
        test_inputs = data['test_inputs']  # (N_test, M)
        test_targets = data['test_targets']  # (N_test,)
        test_powers = data['test_powers']  # (N_test, K)
        
        # Evaluate each baseline
        for baseline_name, baseline_class in AVAILABLE_BASELINES.items():
            try:
                baseline = baseline_class()
                
                # Get predictions
                predictions = []
                for i in range(len(test_inputs)):
                    probe_measurements = test_inputs[i]  # (M,)
                    all_powers = test_powers[i]  # (K,)
                    
                    # Baseline returns scores for all K configurations
                    scores = baseline.score(
                        probe_measurements=probe_measurements,
                        all_powers=all_powers,
                        M=config.system.M,
                        K=config.system.K
                    )
                    predictions.append(scores)
                
                predictions = torch.FloatTensor(np.array(predictions))
                targets = torch.LongTensor(test_targets)
                
                # Compute metrics
                metrics = {}
                for metric_name in config.metrics:
                    metric = get_metric(metric_name)
                    metrics[metric_name] = metric.compute(predictions, targets)
                
                baseline_results[baseline_name] = metrics
                logger.info(f"Baseline {baseline_name}: {metrics}")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate baseline {baseline_name}: {e}")
        
        return baseline_results
    
    def run(
        self,
        config: ExperimentConfig,
        progress_callback: Optional[Callable] = None
    ) -> ExperimentResult:
        """Run a complete experiment.
        
        Args:
            config: Experiment configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            ExperimentResult with all metrics and training history
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            # Setup
            self._set_seed(config.training.random_seed)
            self.device = self._setup_device(config)
            
            # Load/generate data
            raw_data = self._load_data(config)
            
            # Generate probes
            probe_matrix = self._generate_probes(config)
            
            # Apply probes to data
            data = self._apply_probes(raw_data, probe_matrix, config)
            
            # Build model
            model = self._build_model(config)
            model = model.to(self.device)
            model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Setup training
            optimizer, scheduler, criterion = self._setup_training(model, config)
            
            # Create dataloaders
            train_loader, val_loader, test_loader = self._create_dataloaders(data, config)
            
            # Train model
            history, best_epoch, total_epochs = self._train(
                model, train_loader, val_loader,
                optimizer, scheduler, criterion,
                config, self.device, progress_callback
            )
            
            # Evaluate on test set
            test_metrics = self._evaluate(model, test_loader, self.device, config.metrics)
            
            # Evaluate baselines
            baseline_results = self._evaluate_baselines(data, config)
            
            # Prepare result
            training_time = time.time() - start_time
            
            # Get primary metric
            primary_metric_name = 'top_1_accuracy' if 'top_1_accuracy' in config.metrics else config.metrics[0]
            primary_metric_value = test_metrics[primary_metric_name]
            
            result = ExperimentResult(
                config=config,
                metrics=test_metrics,
                training_history=history,
                best_epoch=best_epoch,
                total_epochs=total_epochs,
                training_time_seconds=training_time,
                model_parameters=model_parameters,
                timestamp=timestamp,
                status='completed',
                baseline_results=baseline_results,
                primary_metric_name=primary_metric_name,
                primary_metric_value=primary_metric_value,
            )
            
            # Save to database
            if self.result_tracker:
                self.result_tracker.add_result(result)
            
            logger.info(f"Experiment completed in {training_time:.2f}s")
            logger.info(f"Test metrics: {test_metrics}")
            
            return result
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            
            training_time = time.time() - start_time
            
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
                primary_metric_name='',
                primary_metric_value=0.0,
            )
            
            if self.result_tracker:
                self.result_tracker.add_result(result)
            
            return result
