"""Experiment runner for orchestrating single experiments."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path

from ris_research_engine.foundation import (
    ExperimentConfig, ExperimentResult, SystemConfig, TrainingConfig, ResultTracker
)
from ris_research_engine.plugins.probes import get_probe
from ris_research_engine.plugins.models import get_model
from ris_research_engine.plugins.data_sources import get_data_source
from ris_research_engine.plugins.metrics import (
    TopKAccuracy, HitAtL, MeanReciprocalRank,
    SpectralEfficiency, PowerRatio, InferenceTime
)


class ExperimentRunner:
    """Orchestrates the execution of a single experiment."""
    
    def __init__(self, db_path: str = "outputs/experiments/results.db"):
        """Initialize the experiment runner.
        
        Args:
            db_path: Path to results database
        """
        self.db_path = db_path
        self.tracker = ResultTracker(db_path)
        self._ensure_output_dirs()
    
    def _ensure_output_dirs(self):
        """Ensure output directories exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path("outputs/plots").mkdir(parents=True, exist_ok=True)
        Path("outputs/models").mkdir(parents=True, exist_ok=True)
    
    def run(self, 
            config: ExperimentConfig,
            progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
            campaign_name: Optional[str] = None) -> ExperimentResult:
        """Run a single experiment.
        
        Args:
            config: Experiment configuration
            progress_callback: Optional callback(epoch, total_epochs, metrics)
            campaign_name: Optional campaign name to associate with this experiment
            
        Returns:
            ExperimentResult with metrics and training history
        """
        print(f"Starting experiment: {config.name}")
        start_time = time.time()
        
        try:
            # 1. Load data
            print(f"  Loading data from {config.data_source}...")
            data_source = get_data_source(config.data_source)
            data = data_source.load(config.system, **config.data_params)
            
            # 2. Generate probe measurements
            print(f"  Generating probe measurements with {config.probe_type}...")
            probe = get_probe(config.probe_type)
            probe_matrix = probe.generate(
                N=config.system.N,
                M=config.system.M,
                **config.probe_params
            )
            
            # Extract features based on probe measurements
            train_inputs = self._apply_probe(data['train_inputs'], probe_matrix, config.system)
            val_inputs = self._apply_probe(data['val_inputs'], probe_matrix, config.system)
            test_inputs = self._apply_probe(data['test_inputs'], probe_matrix, config.system)
            
            # 3. Build model
            print(f"  Building {config.model_type} model...")
            model_builder = get_model(config.model_type)
            
            # Input dimension is M * N (probe measurements * elements)
            input_dim = config.system.M * config.system.N
            output_dim = config.system.K  # Number of codebook entries
            
            model = model_builder.build(
                input_dim=input_dim,
                output_dim=output_dim,
                **config.model_params
            )
            
            # Count parameters
            model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Model has {model_parameters:,} trainable parameters")
            
            # 4. Setup training
            device = self._get_device(config.training.device)
            model = model.to(device)
            
            # Create data loaders
            train_loader = self._create_dataloader(
                train_inputs, data['train_targets'],
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=config.training.num_workers
            )
            val_loader = self._create_dataloader(
                val_inputs, data['val_targets'],
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.training.num_workers
            )
            test_loader = self._create_dataloader(
                test_inputs, data['test_targets'],
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.training.num_workers
            )
            
            # Setup optimizer
            optimizer = self._create_optimizer(model, config.training)
            
            # Setup loss function
            criterion = self._create_loss_function(config.training.loss_function)
            
            # Setup scheduler
            scheduler = self._create_scheduler(optimizer, config.training)
            
            # 5. Train model
            print(f"  Training for up to {config.training.max_epochs} epochs...")
            training_history = {
                'train_loss': [],
                'val_loss': [],
                'val_accuracy': [],
            }
            
            best_val_loss = float('inf')
            best_epoch = 0
            patience_counter = 0
            
            for epoch in range(config.training.max_epochs):
                # Train one epoch
                train_loss = self._train_epoch(model, train_loader, optimizer, criterion, device)
                training_history['train_loss'].append(train_loss)
                
                # Validate
                val_loss, val_accuracy = self._validate_epoch(model, val_loader, criterion, device)
                training_history['val_loss'].append(val_loss)
                training_history['val_accuracy'].append(val_accuracy)
                
                # Update scheduler
                if scheduler is not None:
                    if config.training.scheduler == 'plateau':
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                
                # Check for improvement
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
                        'val_accuracy': val_accuracy
                    })
                
                # Print progress
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"    Epoch {epoch+1}/{config.training.max_epochs}: "
                          f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                          f"val_accuracy={val_accuracy:.3f}")
                
                # Early stopping
                if patience_counter >= config.training.early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            
            total_epochs = epoch + 1
            
            # 6. Evaluate on test set
            print(f"  Evaluating on test set...")
            test_metrics = self._evaluate(
                model, test_loader, data['test_targets'],
                data['test_powers'], config, device
            )
            
            # 7. Create result
            training_time = time.time() - start_time
            
            result = ExperimentResult(
                config=config,
                metrics=test_metrics,
                training_history=training_history,
                best_epoch=best_epoch,
                total_epochs=total_epochs,
                training_time_seconds=training_time,
                model_parameters=model_parameters,
                timestamp=datetime.now().isoformat(),
                status='completed',
                primary_metric_name='top_1_accuracy',
                primary_metric_value=test_metrics.get('top_1_accuracy', 0.0)
            )
            
            # Save to database
            exp_id = self.tracker.save_experiment(result, campaign_name=campaign_name)
            print(f"✅ Experiment completed in {training_time:.1f}s (ID: {exp_id})")
            
            return result
            
        except Exception as e:
            # Create failed result
            training_time = time.time() - start_time
            
            result = ExperimentResult(
                config=config,
                metrics={},
                training_history={},
                best_epoch=0,
                total_epochs=0,
                training_time_seconds=training_time,
                model_parameters=0,
                timestamp=datetime.now().isoformat(),
                status='failed',
                error_message=str(e),
                primary_metric_name='top_1_accuracy',
                primary_metric_value=0.0
            )
            
            # Save to database
            self.tracker.save_experiment(result, campaign_name=campaign_name)
            print(f"❌ Experiment failed: {e}")
            
            raise
    
    def _apply_probe(self, inputs: np.ndarray, probe_matrix: np.ndarray, 
                     system: SystemConfig) -> np.ndarray:
        """Apply probe matrix to extract measurements.
        
        Args:
            inputs: Input data (n_samples, M*N)
            probe_matrix: Probe matrix (M, N)
            system: System configuration
            
        Returns:
            Processed inputs (n_samples, M*N)
        """
        # The inputs are already encoded with probe information
        # from the data source. We just need to ensure they match
        # the expected shape.
        return inputs
    
    def _get_device(self, device_str: str) -> torch.device:
        """Get PyTorch device."""
        if device_str == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device_str)
    
    def _create_dataloader(self, inputs: np.ndarray, targets: np.ndarray,
                          batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
        """Create PyTorch DataLoader."""
        inputs_tensor = torch.FloatTensor(inputs)
        targets_tensor = torch.LongTensor(targets)
        dataset = TensorDataset(inputs_tensor, targets_tensor)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True
        )
    
    def _create_optimizer(self, model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
        """Create optimizer."""
        if config.optimizer == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'adamw':
            return optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    def _create_loss_function(self, loss_fn: str) -> nn.Module:
        """Create loss function."""
        if loss_fn == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_fn == 'mse':
            return nn.MSELoss()
        elif loss_fn == 'mae':
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
    
    def _create_scheduler(self, optimizer: optim.Optimizer, 
                         config: TrainingConfig) -> Optional[Any]:
        """Create learning rate scheduler."""
        if config.scheduler == 'none':
            return None
        elif config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.max_epochs
            )
        elif config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=config.max_epochs // 3, gamma=0.1
            )
        elif config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
        else:
            raise ValueError(f"Unknown scheduler: {config.scheduler}")
    
    def _train_epoch(self, model: nn.Module, loader: DataLoader,
                    optimizer: optim.Optimizer, criterion: nn.Module,
                    device: torch.device) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def _validate_epoch(self, model: nn.Module, loader: DataLoader,
                       criterion: nn.Module, device: torch.device) -> tuple:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _evaluate(self, model: nn.Module, loader: DataLoader,
                 targets: np.ndarray, powers: np.ndarray,
                 config: ExperimentConfig, device: torch.device) -> Dict[str, float]:
        """Evaluate model and compute all metrics."""
        model.eval()
        
        # Get predictions
        all_predictions = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                _, predicted = torch.max(outputs, 1)
                all_predictions.append(predicted.cpu().numpy())
        
        predictions = np.concatenate(all_predictions)
        probs = np.concatenate(all_probs)
        
        # Convert to torch tensors for metrics
        probs_tensor = torch.FloatTensor(probs)
        targets_tensor = torch.LongTensor(targets)
        powers_tensor = torch.FloatTensor(powers)
        
        # Compute metrics
        metrics = {}
        
        # Top-k accuracy
        for k in [1, 3, 5]:
            if k <= config.system.K:
                top_k_acc = TopKAccuracy(k=k)
                metrics[f'top_{k}_accuracy'] = top_k_acc.compute(probs_tensor, targets_tensor, powers_tensor)
        
        # Hit@L
        for l in [1, 3, 5]:
            if l <= config.system.K:
                hit_at_l = HitAtL(L=l)  # Use capital L
                metrics[f'hit_at_{l}'] = hit_at_l.compute(probs_tensor, targets_tensor, powers_tensor)
        
        # Mean reciprocal rank
        mrr = MeanReciprocalRank()
        metrics['mean_reciprocal_rank'] = mrr.compute(probs_tensor, targets_tensor, powers_tensor)
        
        # Power ratio - compute from predictions and power matrix
        # Get achieved powers by selecting predicted configurations
        achieved_powers = []
        optimal_powers = []
        for i, pred_idx in enumerate(predictions):
            achieved_powers.append(powers[i, pred_idx])
            optimal_powers.append(powers[i, targets[i]])
        
        achieved_powers_tensor = torch.FloatTensor(achieved_powers)
        optimal_powers_tensor = torch.FloatTensor(optimal_powers)
        
        # Avoid division by zero
        epsilon = 1e-10
        optimal_powers_tensor = optimal_powers_tensor.clamp(min=epsilon)
        power_ratio = (achieved_powers_tensor / optimal_powers_tensor).mean().item()
        metrics['power_ratio'] = power_ratio
        
        # Spectral efficiency - compute from power ratios
        # SE ~ log2(1 + SNR * power_ratio)
        snr_linear = 10 ** (config.system.snr_db / 10.0)
        spectral_eff = torch.log2(1 + snr_linear * achieved_powers_tensor / optimal_powers_tensor).mean().item()
        metrics['spectral_efficiency'] = spectral_eff
        
        # Inference time - measure directly
        # Get a sample input from the loader
        sample_input, _ = next(iter(loader))
        sample_input = sample_input[:1].to(device)  # Take first sample only
        
        # Measure inference time
        model.eval()
        num_warmup = 10
        num_iterations = 100
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(sample_input)
        
        # Synchronize if using CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        import time
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(sample_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        elapsed_time_s = end_time - start_time
        inference_time_ms = (elapsed_time_s / num_iterations) * 1000
        metrics['inference_time_ms'] = inference_time_ms
        
        return metrics
