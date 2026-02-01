"""Interactive dashboard with ipywidgets for RIS Auto-Research Engine."""

import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time

from ris_research_engine.foundation import (
    SystemConfig, TrainingConfig, ExperimentConfig, ResultTracker
)
from ris_research_engine.foundation.logging_config import get_logger
from ris_research_engine.engine import ExperimentRunner, SearchController
from ris_research_engine.plugins.probes import list_probes
from ris_research_engine.plugins.models import list_models

logger = get_logger(__name__)


class RISDashboard:
    """Interactive 5-tab dashboard for RIS experiments."""
    
    def __init__(self, db_path: str = "results.db"):
        """
        Initialize the RIS Dashboard.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.tracker = ResultTracker(db_path)
        self.runner = ExperimentRunner()
        self.controller = SearchController(db_path)
        
        # Experiment queue
        self.queue = []
        
        # Create widgets
        self._create_widgets()
        
        logger.info("RISDashboard initialized")
    
    def _create_widgets(self):
        """Create all widgets for the dashboard."""
        
        # === Tab 1: Configure ===
        # System parameters
        self.N_slider = widgets.IntSlider(value=64, min=16, max=256, step=16, 
                                         description='N (elements):')
        self.K_slider = widgets.IntSlider(value=64, min=16, max=256, step=16,
                                         description='K (codebook):')
        self.M_slider = widgets.IntSlider(value=8, min=2, max=64, step=2,
                                         description='M (measurements):')
        self.freq_slider = widgets.FloatSlider(value=28, min=1, max=100, step=1,
                                              description='Frequency (GHz):')
        self.snr_slider = widgets.FloatSlider(value=20, min=-10, max=40, step=5,
                                             description='SNR (dB):')
        
        # Probe and model selection
        available_probes = list_probes()
        available_models = list_models()
        
        self.probe_dropdown = widgets.Dropdown(
            options=available_probes,
            value=available_probes[0] if available_probes else None,
            description='Probe:'
        )
        
        self.model_dropdown = widgets.Dropdown(
            options=available_models,
            value=available_models[0] if available_models else None,
            description='Model:'
        )
        
        # Training parameters
        self.epochs_slider = widgets.IntSlider(value=100, min=10, max=500, step=10,
                                              description='Epochs:')
        self.lr_dropdown = widgets.Dropdown(
            options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            value=1e-3,
            description='Learning Rate:'
        )
        self.batch_slider = widgets.IntSlider(value=64, min=16, max=256, step=16,
                                             description='Batch Size:')
        
        # Queue management
        self.add_queue_btn = widgets.Button(description='Add to Queue', 
                                           button_style='success')
        self.add_queue_btn.on_click(self._add_to_queue)
        
        self.view_queue_btn = widgets.Button(description='View Queue',
                                            button_style='info')
        self.view_queue_btn.on_click(self._view_queue)
        
        self.clear_queue_btn = widgets.Button(description='Clear Queue',
                                             button_style='warning')
        self.clear_queue_btn.on_click(self._clear_queue)
        
        self.queue_output = widgets.Output()
        
        # === Tab 2: Run ===
        self.run_single_btn = widgets.Button(description='Run Single Experiment',
                                            button_style='primary',
                                            layout=widgets.Layout(width='200px'))
        self.run_single_btn.on_click(self._run_single)
        
        self.run_queue_btn = widgets.Button(description='Run Queue',
                                           button_style='success',
                                           layout=widgets.Layout(width='200px'))
        self.run_queue_btn.on_click(self._run_queue)
        
        self.run_search_btn = widgets.Button(description='Run Search',
                                            button_style='info',
                                            layout=widgets.Layout(width='200px'))
        self.run_search_btn.on_click(self._run_search)
        
        self.stop_btn = widgets.Button(description='Stop',
                                      button_style='danger',
                                      layout=widgets.Layout(width='200px'),
                                      disabled=True)
        
        self.progress_bar = widgets.FloatProgress(value=0, min=0, max=1,
                                                 description='Progress:',
                                                 layout=widgets.Layout(width='100%'))
        
        self.status_text = widgets.HTML(value='<b>Ready</b>')
        
        self.run_output = widgets.Output()
        
        # === Tab 3: Results ===
        self.exp_selector = widgets.Dropdown(
            options=[],
            description='Experiment:',
            layout=widgets.Layout(width='100%')
        )
        self.exp_selector.observe(self._on_experiment_selected, names='value')
        
        self.refresh_exp_btn = widgets.Button(description='Refresh',
                                             button_style='info')
        self.refresh_exp_btn.on_click(self._refresh_experiments)
        
        self.results_output = widgets.Output()
        
        # === Tab 4: Analysis ===
        self.campaign_selector = widgets.Dropdown(
            options=[],
            description='Campaign:',
            layout=widgets.Layout(width='100%')
        )
        self.campaign_selector.observe(self._on_campaign_selected, names='value')
        
        self.refresh_camp_btn = widgets.Button(description='Refresh',
                                              button_style='info')
        self.refresh_camp_btn.on_click(self._refresh_campaigns)
        
        self.metric_dropdown = widgets.Dropdown(
            options=['top_1_accuracy', 'top_5_accuracy', 'top_10_accuracy',
                    'mean_reciprocal_rank', 'inference_time'],
            value='top_1_accuracy',
            description='Metric:'
        )
        
        self.plot_comparison_btn = widgets.Button(description='Plot Comparison',
                                                 button_style='primary')
        self.plot_comparison_btn.on_click(self._plot_comparison)
        
        self.export_btn = widgets.Button(description='Export Results',
                                        button_style='success')
        self.export_btn.on_click(self._export_results)
        
        self.analysis_output = widgets.Output()
        
        # === Tab 5: Database ===
        self.filter_campaign = widgets.Text(description='Campaign:',
                                           placeholder='Filter by campaign')
        self.filter_status = widgets.Dropdown(
            options=['all', 'completed', 'failed', 'pruned'],
            value='all',
            description='Status:'
        )
        self.filter_limit = widgets.IntSlider(value=20, min=5, max=100, step=5,
                                             description='Limit:')
        
        self.apply_filter_btn = widgets.Button(description='Apply Filter',
                                              button_style='primary')
        self.apply_filter_btn.on_click(self._apply_filter)
        
        self.export_csv_btn = widgets.Button(description='Export CSV',
                                            button_style='success')
        self.export_csv_btn.on_click(lambda x: self._export_database('csv'))
        
        self.export_json_btn = widgets.Button(description='Export JSON',
                                             button_style='info')
        self.export_json_btn.on_click(lambda x: self._export_database('json'))
        
        self.db_output = widgets.Output()
        
        # Initialize
        self._refresh_experiments(None)
        self._refresh_campaigns(None)
    
    def _add_to_queue(self, btn):
        """Add current configuration to queue."""
        config = self._get_current_config()
        self.queue.append(config)
        
        with self.queue_output:
            clear_output(wait=True)
            print(f"✓ Added to queue (total: {len(self.queue)})")
            print(f"Config: {config['name']}")
    
    def _view_queue(self, btn):
        """Display current queue."""
        with self.queue_output:
            clear_output(wait=True)
            if not self.queue:
                print("Queue is empty")
            else:
                print(f"Queue ({len(self.queue)} experiments):")
                print("=" * 60)
                for i, cfg in enumerate(self.queue, 1):
                    print(f"{i}. {cfg['name']}")
    
    def _clear_queue(self, btn):
        """Clear the experiment queue."""
        self.queue.clear()
        with self.queue_output:
            clear_output(wait=True)
            print("Queue cleared")
    
    def _get_current_config(self) -> ExperimentConfig:
        """Build ExperimentConfig from current widget values."""
        N = self.N_slider.value
        N_x = int(np.sqrt(N))
        N_y = N_x
        
        system = SystemConfig(
            N=N, N_x=N_x, N_y=N_y,
            K=self.K_slider.value,
            M=self.M_slider.value,
            frequency=self.freq_slider.value * 1e9,
            snr_db=self.snr_slider.value
        )
        
        training = TrainingConfig(
            learning_rate=self.lr_dropdown.value,
            batch_size=self.batch_slider.value,
            max_epochs=self.epochs_slider.value
        )
        
        probe = self.probe_dropdown.value
        model = self.model_dropdown.value
        
        config = ExperimentConfig(
            name=f"{probe}_{model}_M{system.M}_K{system.K}",
            system=system,
            training=training,
            probe_type=probe,
            probe_params={},
            model_type=model,
            model_params={},
            data_source='synthetic_rayleigh',
            data_params={'n_samples': 1000},
            metrics=['top_k_accuracy', 'mean_reciprocal_rank'],
            tags=['dashboard']
        )
        
        return config
    
    def _run_single(self, btn):
        """Run a single experiment."""
        config = self._get_current_config()
        
        self.run_single_btn.disabled = True
        self.run_queue_btn.disabled = True
        self.run_search_btn.disabled = True
        self.stop_btn.disabled = False
        
        with self.run_output:
            clear_output(wait=True)
            print(f"Running: {config.name}")
            print("=" * 60)
            
            try:
                self.status_text.value = '<b>Running...</b>'
                self.progress_bar.value = 0.5
                
                result = self.runner.run(config)
                exp_id = self.tracker.save_experiment(result)
                
                self.progress_bar.value = 1.0
                self.status_text.value = f'<b>Completed - ID: {exp_id}</b>'
                
                print(f"✓ Experiment completed - ID: {exp_id}")
                print(f"Status: {result.status}")
                print(f"Top-1 Accuracy: {result.metrics.get('top_1_accuracy', 0):.4f}")
                
                self._refresh_experiments(None)
                
            except Exception as e:
                self.status_text.value = '<b>Failed</b>'
                print(f"✗ Error: {str(e)}")
                logger.error(f"Experiment failed: {e}", exc_info=True)
            
            finally:
                self.run_single_btn.disabled = False
                self.run_queue_btn.disabled = False
                self.run_search_btn.disabled = False
                self.stop_btn.disabled = True
                self.progress_bar.value = 0
    
    def _run_queue(self, btn):
        """Run all experiments in queue."""
        if not self.queue:
            with self.run_output:
                clear_output(wait=True)
                print("Queue is empty")
            return
        
        self.run_single_btn.disabled = True
        self.run_queue_btn.disabled = True
        self.run_search_btn.disabled = True
        self.stop_btn.disabled = False
        
        with self.run_output:
            clear_output(wait=True)
            print(f"Running queue ({len(self.queue)} experiments)...")
            print("=" * 60)
            
            total = len(self.queue)
            
            for i, config in enumerate(self.queue, 1):
                print(f"\n[{i}/{total}] {config.name}")
                
                try:
                    self.status_text.value = f'<b>Running {i}/{total}</b>'
                    self.progress_bar.value = i / total
                    
                    result = self.runner.run(config)
                    exp_id = self.tracker.save_experiment(result)
                    
                    print(f"  ✓ Completed - ID: {exp_id}")
                    
                except Exception as e:
                    print(f"  ✗ Failed: {str(e)}")
                    logger.error(f"Queue experiment failed: {e}")
            
            print("\n" + "=" * 60)
            print("Queue completed")
            
            self.queue.clear()
            self._refresh_experiments(None)
        
        self.run_single_btn.disabled = False
        self.run_queue_btn.disabled = False
        self.run_search_btn.disabled = False
        self.stop_btn.disabled = True
        self.progress_bar.value = 0
        self.status_text.value = '<b>Ready</b>'
    
    def _run_search(self, btn):
        """Run a search campaign."""
        with self.run_output:
            clear_output(wait=True)
            print("Search feature coming soon...")
            print("Use RISEngine.search() for now")
    
    def _on_experiment_selected(self, change):
        """Handle experiment selection."""
        exp_id = change['new']
        if exp_id is None:
            return
        
        with self.results_output:
            clear_output(wait=True)
            
            exp = self.tracker.get_experiment(exp_id)
            if not exp:
                print("Experiment not found")
                return
            
            # Display metrics
            print(f"Experiment ID: {exp['id']}")
            print(f"Name: {exp['name']}")
            print(f"Status: {exp['status']}")
            print("=" * 70)
            
            print("\nMetrics:")
            metrics_data = []
            for key, val in exp['metrics'].items():
                if isinstance(val, float):
                    metrics_data.append({'Metric': key, 'Value': f"{val:.4f}"})
                else:
                    metrics_data.append({'Metric': key, 'Value': str(val)})
            
            df = pd.DataFrame(metrics_data)
            print(df.to_string(index=False))
            
            print(f"\nTraining Time: {exp['training_time_seconds']:.2f}s")
            print(f"Model Parameters: {exp['model_parameters']:,}")
            
            # Plot training curves
            if 'training_history' in exp and exp['training_history']:
                history = exp['training_history']
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Loss
                if 'train_loss' in history:
                    epochs = range(1, len(history['train_loss']) + 1)
                    axes[0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
                    if 'val_loss' in history:
                        axes[0].plot(epochs, history['val_loss'], label='Val', linewidth=2)
                    axes[0].set_xlabel('Epoch')
                    axes[0].set_ylabel('Loss')
                    axes[0].set_title('Training Loss')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                
                # Accuracy
                if 'val_top_1_accuracy' in history:
                    epochs = range(1, len(history['val_top_1_accuracy']) + 1)
                    axes[1].plot(epochs, history['val_top_1_accuracy'], 
                               label='Top-1', linewidth=2)
                    if 'val_top_5_accuracy' in history:
                        axes[1].plot(epochs, history['val_top_5_accuracy'], 
                                   label='Top-5', linewidth=2)
                    axes[1].set_xlabel('Epoch')
                    axes[1].set_ylabel('Accuracy')
                    axes[1].set_title('Validation Accuracy')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
    
    def _refresh_experiments(self, btn):
        """Refresh experiment list."""
        experiments = self.tracker.get_all_experiments(limit=50)
        
        options = [(f"ID {e['id']}: {e['name'][:40]}", e['id']) 
                  for e in experiments]
        
        self.exp_selector.options = options
        
        if btn is not None:
            with self.results_output:
                clear_output(wait=True)
                print(f"Refreshed - {len(experiments)} experiments found")
    
    def _on_campaign_selected(self, change):
        """Handle campaign selection."""
        campaign_id = change['new']
        if campaign_id is None:
            return
        
        with self.analysis_output:
            clear_output(wait=True)
            
            campaign = self.tracker.get_campaign(campaign_id=campaign_id)
            if not campaign:
                print("Campaign not found")
                return
            
            print(f"Campaign: {campaign['name']}")
            print(f"Strategy: {campaign['search_strategy']}")
            print("=" * 70)
            print(f"Total Experiments: {campaign['total_experiments']}")
            print(f"Completed: {campaign['completed']}")
            
            # Get campaign experiments
            experiments = self.tracker.get_all_experiments(campaign_name=campaign['name'])
            
            if experiments:
                completed = [e for e in experiments if e['status'] == 'completed']
                
                if completed:
                    # Best result
                    best = max(completed, key=lambda x: x['metrics'].get('top_1_accuracy', 0))
                    print(f"\nBest Result:")
                    print(f"  {best['name']}")
                    print(f"  Top-1 Accuracy: {best['metrics'].get('top_1_accuracy', 0):.4f}")
    
    def _refresh_campaigns(self, btn):
        """Refresh campaign list."""
        # Get all campaigns by querying database directly
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, name FROM campaigns ORDER BY timestamp DESC LIMIT 50")
            rows = cursor.fetchall()
            campaigns = [dict(row) for row in rows]
            conn.close()
        except sqlite3.Error:
            campaigns = []
        
        options = [(f"ID {c['id']}: {c['name'][:40]}", c['id']) 
                  for c in campaigns]
        
        self.campaign_selector.options = options
        
        if btn is not None:
            with self.analysis_output:
                clear_output(wait=True)
                print(f"Refreshed - {len(campaigns)} campaigns found")
    
    def _plot_comparison(self, btn):
        """Plot comparison for selected campaign."""
        campaign_id = self.campaign_selector.value
        if campaign_id is None:
            with self.analysis_output:
                print("No campaign selected")
            return
        
        metric = self.metric_dropdown.value
        
        with self.analysis_output:
            clear_output(wait=True)
            
            campaign = self.tracker.get_campaign(campaign_id=campaign_id)
            experiments = self.tracker.get_all_experiments(campaign_name=campaign['name'])
            completed = [e for e in experiments if e['status'] == 'completed']
            
            if not completed:
                print("No completed experiments in campaign")
                return
            
            # Bar chart
            data = [(e['name'][:20], e['metrics'].get(metric, 0)) 
                   for e in completed]
            names, values = zip(*data)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            bars = ax.bar(range(len(names)), values)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{campaign["name"]} - {metric.replace("_", " ").title()}')
            ax.grid(True, axis='y', alpha=0.3)
            
            # Color bars
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.tight_layout()
            plt.show()
    
    def _export_results(self, btn):
        """Export analysis results."""
        campaign_id = self.campaign_selector.value
        if campaign_id is None:
            with self.analysis_output:
                print("No campaign selected")
            return
        
        with self.analysis_output:
            campaign = self.tracker.get_campaign(campaign_id=campaign_id)
            experiments = self.tracker.get_all_experiments(campaign_name=campaign['name'])
            
            # Create DataFrame
            data = []
            for exp in experiments:
                row = {
                    'id': exp['id'],
                    'name': exp['name'],
                    'probe': exp['probe_type'],
                    'model': exp['model_type'],
                    'M': exp['M'],
                    'K': exp['K'],
                    'status': exp['status']
                }
                row.update(exp['metrics'])
                data.append(row)
            
            df = pd.DataFrame(data)
            
            filename = f"{campaign['name']}_results.csv"
            df.to_csv(filename, index=False)
            
            print(f"✓ Exported to {filename}")
    
    def _apply_filter(self, btn):
        """Apply database filters."""
        with self.db_output:
            clear_output(wait=True)
            
            campaign = self.filter_campaign.value or None
            status = self.filter_status.value if self.filter_status.value != 'all' else None
            limit = self.filter_limit.value
            
            experiments = self.tracker.get_all_experiments(
                campaign_name=campaign,
                status=status,
                limit=limit
            )
            
            # Display as table
            if not experiments:
                print("No experiments found")
                return
            
            data = []
            for exp in experiments:
                data.append({
                    'ID': exp['id'],
                    'Name': exp['name'][:30],
                    'Probe': exp['probe_type'],
                    'Model': exp['model_type'],
                    'M': exp['M'],
                    'K': exp['K'],
                    'Top-1': f"{exp['metrics'].get('top_1_accuracy', 0):.3f}",
                    'Status': exp['status']
                })
            
            df = pd.DataFrame(data)
            print(df.to_string(index=False))
            print(f"\nShowing {len(df)} of {len(experiments)} experiments")
    
    def _export_database(self, format: str):
        """Export database to CSV or JSON."""
        with self.db_output:
            experiments = self.tracker.get_all_experiments(limit=1000)
            
            if format == 'csv':
                data = []
                for exp in experiments:
                    row = {
                        'id': exp['id'],
                        'name': exp['name'],
                        'probe': exp['probe_type'],
                        'model': exp['model_type'],
                        'M': exp['M'],
                        'K': exp['K'],
                        'status': exp['status'],
                        'timestamp': exp['timestamp']
                    }
                    row.update(exp['metrics'])
                    data.append(row)
                
                df = pd.DataFrame(data)
                filename = f"experiments_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False)
                print(f"✓ Exported {len(df)} experiments to {filename}")
                
            elif format == 'json':
                filename = f"experiments_{time.strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(experiments, f, indent=2)
                print(f"✓ Exported {len(experiments)} experiments to {filename}")
    
    def display(self):
        """Display the dashboard interface."""
        # Tab 1: Configure
        tab1 = widgets.VBox([
            widgets.HTML('<h3>System Parameters</h3>'),
            self.N_slider,
            self.K_slider,
            self.M_slider,
            self.freq_slider,
            self.snr_slider,
            widgets.HTML('<h3>Algorithm Selection</h3>'),
            self.probe_dropdown,
            self.model_dropdown,
            widgets.HTML('<h3>Training Parameters</h3>'),
            self.epochs_slider,
            self.lr_dropdown,
            self.batch_slider,
            widgets.HTML('<h3>Queue Management</h3>'),
            widgets.HBox([self.add_queue_btn, self.view_queue_btn, self.clear_queue_btn]),
            self.queue_output
        ])
        
        # Tab 2: Run
        tab2 = widgets.VBox([
            widgets.HTML('<h3>Run Experiments</h3>'),
            widgets.HBox([self.run_single_btn, self.run_queue_btn, 
                         self.run_search_btn, self.stop_btn]),
            self.progress_bar,
            self.status_text,
            widgets.HTML('<hr>'),
            self.run_output
        ])
        
        # Tab 3: Results
        tab3 = widgets.VBox([
            widgets.HTML('<h3>Experiment Results</h3>'),
            widgets.HBox([self.exp_selector, self.refresh_exp_btn]),
            widgets.HTML('<hr>'),
            self.results_output
        ])
        
        # Tab 4: Analysis
        tab4 = widgets.VBox([
            widgets.HTML('<h3>Campaign Analysis</h3>'),
            widgets.HBox([self.campaign_selector, self.refresh_camp_btn]),
            self.metric_dropdown,
            widgets.HBox([self.plot_comparison_btn, self.export_btn]),
            widgets.HTML('<hr>'),
            self.analysis_output
        ])
        
        # Tab 5: Database
        tab5 = widgets.VBox([
            widgets.HTML('<h3>Database Explorer</h3>'),
            widgets.HBox([self.filter_campaign, self.filter_status, self.filter_limit]),
            self.apply_filter_btn,
            widgets.HBox([self.export_csv_btn, self.export_json_btn]),
            widgets.HTML('<hr>'),
            self.db_output
        ])
        
        # Create tabs
        tabs = widgets.Tab(children=[tab1, tab2, tab3, tab4, tab5])
        tabs.set_title(0, 'Configure')
        tabs.set_title(1, 'Run')
        tabs.set_title(2, 'Results')
        tabs.set_title(3, 'Analysis')
        tabs.set_title(4, 'Database')
        
        # Display
        display(widgets.VBox([
            widgets.HTML('<h2>🔬 RIS Auto-Research Engine Dashboard</h2>'),
            tabs
        ]))
