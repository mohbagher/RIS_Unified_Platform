"""Interactive Jupyter dashboard for the RIS Auto-Research Engine."""

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import threading
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

from ris_research_engine.foundation import (
    SystemConfig, TrainingConfig, ExperimentConfig, ResultTracker
)
from ris_research_engine.engine import ExperimentRunner, ResultAnalyzer
from ris_research_engine.plugins.probes import list_probes
from ris_research_engine.plugins.models import list_models, get_model


class RISDashboard:
    """Interactive 5-tab dashboard for experiment management."""
    
    def __init__(self, db_path: str = "outputs/experiments/results.db"):
        """Initialize the dashboard.
        
        Args:
            db_path: Path to results database
        """
        self.runner = ExperimentRunner(db_path)
        self.tracker = ResultTracker(db_path)
        self.analyzer = ResultAnalyzer(self.tracker)
        self.queue = []
        self.stop_flag = False
        self.current_experiment_id = None
        
        # Build all widgets
        self._build_widgets()
        self._build_tabs()
    
    def display(self):
        """Display the dashboard."""
        return display(self.tabs)
    
    def _build_widgets(self):
        """Build all widgets for the dashboard."""
        # === Tab 1: Configure ===
        
        # System parameters
        self.n_slider = widgets.IntSlider(value=64, min=16, max=256, step=16, description='N (Elements):')
        self.k_slider = widgets.IntSlider(value=64, min=8, max=256, step=8, description='K (Codebook):')
        self.m_slider = widgets.IntSlider(value=8, min=1, max=64, step=1, description='M (Probes):')
        self.freq_text = widgets.FloatText(value=28e9, description='Frequency (Hz):', style={'description_width': 'initial'})
        self.snr_slider = widgets.IntSlider(value=20, min=0, max=40, step=5, description='SNR (dB):')
        
        # Probe selection
        self.probe_dropdown = widgets.Dropdown(
            options=list_probes(),
            description='Probe Type:',
            style={'description_width': 'initial'}
        )
        
        # Model selection
        self.model_dropdown = widgets.Dropdown(
            options=list_models(),
            description='Model Type:',
            style={'description_width': 'initial'}
        )
        self.model_dropdown.observe(self._on_model_change, names='value')
        
        # Model parameters (dynamic based on selected model)
        self.model_params_output = widgets.Output()
        self.model_param_widgets = {}
        
        # Training parameters
        self.lr_text = widgets.FloatText(value=1e-3, description='Learning Rate:', style={'description_width': 'initial'})
        self.epochs_text = widgets.IntText(value=100, description='Max Epochs:', style={'description_width': 'initial'})
        self.batch_size_text = widgets.IntText(value=64, description='Batch Size:', style={'description_width': 'initial'})
        
        # Data source
        self.data_source_dropdown = widgets.Dropdown(
            options=['synthetic_rayleigh', 'synthetic_rician', 'hdf5_loader'],
            value='synthetic_rayleigh',
            description='Data Source:',
            style={'description_width': 'initial'}
        )
        self.data_path_text = widgets.Text(
            value='',
            placeholder='/path/to/data.h5',
            description='HDF5 Path:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        # Queue management
        self.add_queue_button = widgets.Button(description='Add to Queue', button_style='success')
        self.add_queue_button.on_click(self._add_to_queue)
        
        self.queue_output = widgets.Output(layout=widgets.Layout(height='200px', overflow_y='auto'))
        self.clear_queue_button = widgets.Button(description='Clear Queue', button_style='warning')
        self.clear_queue_button.on_click(self._clear_queue)
        
        # === Tab 2: Run ===
        
        # Run buttons
        self.run_single_button = widgets.Button(description='Run Single', button_style='primary', icon='play')
        self.run_single_button.on_click(self._run_single)
        
        self.run_queue_button = widgets.Button(description='Run Queue', button_style='primary', icon='list')
        self.run_queue_button.on_click(self._run_queue)
        
        self.stop_button = widgets.Button(description='Stop', button_style='danger', icon='stop')
        self.stop_button.on_click(self._stop_execution)
        self.stop_button.disabled = True
        
        # Progress display
        self.current_exp_label = widgets.Label(value='No experiment running')
        self.epoch_progress = widgets.IntProgress(min=0, max=100, description='Epoch:', bar_style='info')
        self.metrics_html = widgets.HTML(value='<i>Metrics will appear here during training...</i>')
        self.campaign_progress = widgets.IntProgress(min=0, max=100, description='Campaign:', bar_style='success')
        
        self.run_output = widgets.Output(layout=widgets.Layout(height='300px', overflow_y='auto'))
        
        # === Tab 3: Results ===
        
        self.experiment_selector = widgets.Dropdown(
            options=[],
            description='Experiment:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='600px')
        )
        self.experiment_selector.observe(self._on_experiment_selected, names='value')
        
        self.refresh_exp_button = widgets.Button(description='Refresh', button_style='info', icon='refresh')
        self.refresh_exp_button.on_click(self._refresh_experiments)
        
        self.metrics_table_html = widgets.HTML(value='<i>Select an experiment to view metrics</i>')
        self.training_plot_output = widgets.Output()
        
        self.compare_selector = widgets.SelectMultiple(
            options=[],
            description='Select to Compare:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='600px', height='150px')
        )
        self.compare_button = widgets.Button(description='Compare Selected', button_style='primary')
        self.compare_button.on_click(self._compare_experiments)
        self.compare_output = widgets.Output()
        
        # === Tab 4: Analysis ===
        
        self.campaign_selector = widgets.Dropdown(
            options=[],
            description='Campaign:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.refresh_campaign_button = widgets.Button(description='Refresh', button_style='info', icon='refresh')
        self.refresh_campaign_button.on_click(self._refresh_campaigns)
        
        self.analysis_output = widgets.Output(layout=widgets.Layout(height='600px', overflow_y='auto'))
        
        self.generate_report_button = widgets.Button(description='Generate Report', button_style='success')
        self.generate_report_button.on_click(self._generate_report)
        
        # === Tab 5: Database ===
        
        self.filter_probe_text = widgets.Text(placeholder='Filter by probe type', description='Probe:')
        self.filter_model_text = widgets.Text(placeholder='Filter by model type', description='Model:')
        self.filter_min_acc_text = widgets.FloatText(value=0.0, description='Min Accuracy:', style={'description_width': 'initial'})
        
        self.sort_dropdown = widgets.Dropdown(
            options=['timestamp', 'top_1_accuracy', 'training_time_seconds'],
            value='timestamp',
            description='Sort by:',
            style={'description_width': 'initial'}
        )
        
        self.apply_filter_button = widgets.Button(description='Apply Filters', button_style='primary')
        self.apply_filter_button.on_click(self._apply_filters)
        
        self.db_table_output = widgets.Output(layout=widgets.Layout(height='400px', overflow_y='auto'))
        
        self.export_csv_button = widgets.Button(description='Export CSV', button_style='info')
        self.export_csv_button.on_click(self._export_csv)
        
        self.export_json_button = widgets.Button(description='Export JSON', button_style='info')
        self.export_json_button.on_click(self._export_json)
        
        self.db_stats_html = widgets.HTML(value='<i>Loading database statistics...</i>')
        
        # Initialize model params display
        self._on_model_change(None)
    
    def _build_tabs(self):
        """Build the 5-tab interface."""
        # Tab 1: Configure
        tab1 = widgets.VBox([
            widgets.HTML('<h3>System Parameters</h3>'),
            widgets.HBox([self.n_slider, self.k_slider, self.m_slider]),
            widgets.HBox([self.freq_text, self.snr_slider]),
            
            widgets.HTML('<h3>Probe & Model Selection</h3>'),
            widgets.HBox([self.probe_dropdown, self.model_dropdown]),
            self.model_params_output,
            
            widgets.HTML('<h3>Training Parameters</h3>'),
            widgets.HBox([self.lr_text, self.epochs_text, self.batch_size_text]),
            
            widgets.HTML('<h3>Data Source</h3>'),
            widgets.HBox([self.data_source_dropdown, self.data_path_text]),
            
            widgets.HTML('<h3>Experiment Queue</h3>'),
            self.add_queue_button,
            self.queue_output,
            self.clear_queue_button,
        ], layout=widgets.Layout(padding='10px'))
        
        # Tab 2: Run
        tab2 = widgets.VBox([
            widgets.HTML('<h3>Run Experiments</h3>'),
            widgets.HBox([self.run_single_button, self.run_queue_button, self.stop_button]),
            
            widgets.HTML('<h3>Progress</h3>'),
            self.current_exp_label,
            self.epoch_progress,
            self.metrics_html,
            self.campaign_progress,
            
            widgets.HTML('<h3>Output Log</h3>'),
            self.run_output,
        ], layout=widgets.Layout(padding='10px'))
        
        # Tab 3: Results
        tab3 = widgets.VBox([
            widgets.HTML('<h3>Select Experiment</h3>'),
            widgets.HBox([self.experiment_selector, self.refresh_exp_button]),
            
            widgets.HTML('<h3>Metrics Summary</h3>'),
            self.metrics_table_html,
            
            widgets.HTML('<h3>Training Curves</h3>'),
            self.training_plot_output,
            
            widgets.HTML('<h3>Compare Experiments</h3>'),
            self.compare_selector,
            self.compare_button,
            self.compare_output,
        ], layout=widgets.Layout(padding='10px'))
        
        # Tab 4: Analysis
        tab4 = widgets.VBox([
            widgets.HTML('<h3>Campaign Analysis</h3>'),
            widgets.HBox([self.campaign_selector, self.refresh_campaign_button]),
            self.generate_report_button,
            
            widgets.HTML('<h3>Analysis Plots</h3>'),
            self.analysis_output,
        ], layout=widgets.Layout(padding='10px'))
        
        # Tab 5: Database
        tab5 = widgets.VBox([
            widgets.HTML('<h3>Filter & Sort</h3>'),
            widgets.HBox([self.filter_probe_text, self.filter_model_text, self.filter_min_acc_text]),
            widgets.HBox([self.sort_dropdown, self.apply_filter_button]),
            
            widgets.HTML('<h3>Experiment Table</h3>'),
            self.db_table_output,
            
            widgets.HTML('<h3>Export & Statistics</h3>'),
            widgets.HBox([self.export_csv_button, self.export_json_button]),
            self.db_stats_html,
        ], layout=widgets.Layout(padding='10px'))
        
        # Create tabs
        self.tabs = widgets.Tab(children=[tab1, tab2, tab3, tab4, tab5])
        self.tabs.set_title(0, 'Configure')
        self.tabs.set_title(1, 'Run')
        self.tabs.set_title(2, 'Results')
        self.tabs.set_title(3, 'Analysis')
        self.tabs.set_title(4, 'Database')
        
        # Initialize displays
        self._refresh_experiments(None)
        self._refresh_campaigns(None)
        self._update_db_stats()
        self._apply_filters(None)
    
    def _on_model_change(self, change):
        """Update model parameters when model selection changes."""
        model_name = self.model_dropdown.value
        
        with self.model_params_output:
            clear_output(wait=True)
            
            # Get default params for selected model
            model_builder = get_model(model_name)
            default_params = model_builder.get_default_params()
            
            # Create widgets for each parameter
            self.model_param_widgets = {}
            param_widgets = []
            
            for param_name, param_value in default_params.items():
                if isinstance(param_value, bool):
                    widget = widgets.Checkbox(value=param_value, description=param_name)
                elif isinstance(param_value, int):
                    widget = widgets.IntText(value=param_value, description=param_name)
                elif isinstance(param_value, float):
                    widget = widgets.FloatText(value=param_value, description=param_name)
                elif isinstance(param_value, str):
                    widget = widgets.Text(value=param_value, description=param_name)
                elif isinstance(param_value, list):
                    widget = widgets.Text(value=str(param_value), description=param_name)
                else:
                    continue
                
                self.model_param_widgets[param_name] = widget
                param_widgets.append(widget)
            
            if param_widgets:
                display(widgets.HTML('<b>Model Parameters:</b>'))
                display(widgets.VBox(param_widgets))
            else:
                display(widgets.HTML('<i>No configurable parameters</i>'))
    
    def _get_current_config(self) -> ExperimentConfig:
        """Build ExperimentConfig from current widget values."""
        # Extract model params
        model_params = {}
        for param_name, widget in self.model_param_widgets.items():
            value = widget.value
            # Try to parse lists
            if isinstance(value, str) and value.startswith('['):
                try:
                    import ast
                    value = ast.literal_eval(value)
                except:
                    pass
            model_params[param_name] = value
        
        # Build system config
        N = self.n_slider.value
        system = SystemConfig(
            N=N,
            N_x=int(N**0.5),
            N_y=int(N**0.5),
            K=self.k_slider.value,
            M=self.m_slider.value,
            frequency=self.freq_text.value,
            snr_db=self.snr_slider.value,
        )
        
        # Build training config
        training = TrainingConfig(
            learning_rate=self.lr_text.value,
            batch_size=self.batch_size_text.value,
            max_epochs=self.epochs_text.value,
        )
        
        # Build data params
        data_params = {'n_train': 10000, 'n_val': 2000, 'n_test': 2000}
        if self.data_source_dropdown.value == 'hdf5_loader' and self.data_path_text.value:
            data_params = {'h5_path': self.data_path_text.value}
        
        # Create experiment config
        name = f"{self.probe_dropdown.value}_{self.model_dropdown.value}_M{self.m_slider.value}_K{self.k_slider.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config = ExperimentConfig(
            name=name,
            system=system,
            training=training,
            probe_type=self.probe_dropdown.value,
            probe_params={},
            model_type=self.model_dropdown.value,
            model_params=model_params,
            data_source=self.data_source_dropdown.value,
            data_params=data_params,
            metrics=['top_1_accuracy', 'hit_at_1', 'power_ratio'],
            tags=['dashboard'],
            notes='Created from dashboard',
            data_fidelity='synthetic',
        )
        
        return config
    
    def _add_to_queue(self, button):
        """Add current configuration to queue."""
        try:
            config = self._get_current_config()
            self.queue.append(config)
            
            with self.queue_output:
                clear_output(wait=True)
                print(f"Experiment Queue ({len(self.queue)} items):")
                print("="*60)
                for i, cfg in enumerate(self.queue, 1):
                    print(f"{i}. {cfg.name}")
                    print(f"   Probe: {cfg.probe_type}, Model: {cfg.model_type}, M={cfg.system.M}, K={cfg.system.K}")
                print("="*60)
        except Exception as e:
            with self.queue_output:
                print(f"Error adding to queue: {e}")
    
    def _clear_queue(self, button):
        """Clear the experiment queue."""
        self.queue = []
        with self.queue_output:
            clear_output(wait=True)
            print("Queue cleared")
    
    def _run_single(self, button):
        """Run single experiment from current config."""
        def run():
            try:
                config = self._get_current_config()
                
                with self.run_output:
                    print(f"\n{'='*60}")
                    print(f"Starting experiment: {config.name}")
                    print(f"{'='*60}")
                
                # Disable buttons
                self.run_single_button.disabled = True
                self.run_queue_button.disabled = True
                self.stop_button.disabled = False
                self.stop_flag = False
                
                # Run experiment
                result = self.runner.run(config, progress_callback=self._update_progress)
                
                with self.run_output:
                    print(f"\n✅ Experiment completed!")
                    print(f"Top-1 Accuracy: {result.metrics.get('top_1_accuracy', 0.0):.3f}")
                    print(f"Training time: {result.training_time_seconds:.1f}s")
                
            except Exception as e:
                with self.run_output:
                    print(f"\n❌ Error: {e}")
            finally:
                # Re-enable buttons
                self.run_single_button.disabled = False
                self.run_queue_button.disabled = False
                self.stop_button.disabled = True
                
                # Reset progress
                self.current_exp_label.value = 'No experiment running'
                self.epoch_progress.value = 0
                self.metrics_html.value = '<i>Completed</i>'
        
        # Run in thread
        thread = threading.Thread(target=run)
        thread.start()
    
    def _run_queue(self, button):
        """Run all queued experiments."""
        def run():
            try:
                if not self.queue:
                    with self.run_output:
                        print("Queue is empty!")
                    return
                
                # Disable buttons
                self.run_single_button.disabled = True
                self.run_queue_button.disabled = True
                self.stop_button.disabled = False
                self.stop_flag = False
                
                # Run all experiments
                total = len(self.queue)
                self.campaign_progress.max = total
                
                for i, config in enumerate(self.queue):
                    if self.stop_flag:
                        with self.run_output:
                            print("\n⚠️ Stopped by user")
                        break
                    
                    with self.run_output:
                        print(f"\n{'='*60}")
                        print(f"[{i+1}/{total}] Running: {config.name}")
                        print(f"{'='*60}")
                    
                    self.campaign_progress.value = i
                    
                    try:
                        result = self.runner.run(config, progress_callback=self._update_progress, campaign_name="dashboard_queue")
                        
                        with self.run_output:
                            print(f"✅ Completed: {result.metrics.get('top_1_accuracy', 0.0):.3f} accuracy")
                    except Exception as e:
                        with self.run_output:
                            print(f"❌ Failed: {e}")
                
                self.campaign_progress.value = total
                
                with self.run_output:
                    print(f"\n{'='*60}")
                    print("Queue execution completed!")
                    print(f"{'='*60}")
                
                # Clear queue
                self.queue = []
                with self.queue_output:
                    clear_output(wait=True)
                    print("Queue cleared after execution")
                
            except Exception as e:
                with self.run_output:
                    print(f"\n❌ Error: {e}")
            finally:
                # Re-enable buttons
                self.run_single_button.disabled = False
                self.run_queue_button.disabled = False
                self.stop_button.disabled = True
                
                # Reset progress
                self.current_exp_label.value = 'No experiment running'
                self.epoch_progress.value = 0
                self.campaign_progress.value = 0
        
        # Run in thread
        thread = threading.Thread(target=run)
        thread.start()
    
    def _stop_execution(self, button):
        """Stop the current execution."""
        self.stop_flag = True
        with self.run_output:
            print("\n⚠️ Stop requested...")
    
    def _update_progress(self, epoch: int, total_epochs: int, metrics: Dict[str, float]):
        """Update progress widgets."""
        self.epoch_progress.max = total_epochs
        self.epoch_progress.value = epoch
        
        metrics_html = f"<b>Epoch {epoch}/{total_epochs}</b><br>"
        for key, value in metrics.items():
            metrics_html += f"{key}: {value:.4f}<br>"
        
        self.metrics_html.value = metrics_html
    
    def _refresh_experiments(self, button):
        """Refresh experiment list."""
        experiments = self.tracker.get_all_experiments(status='completed')
        
        if experiments:
            options = [(f"{e['name']} (ID: {e['id']})", e['id']) for e in experiments]
            self.experiment_selector.options = options
            self.compare_selector.options = options
        else:
            self.experiment_selector.options = [('No experiments found', None)]
            self.compare_selector.options = []
    
    def _on_experiment_selected(self, change):
        """Handle experiment selection."""
        exp_id = change['new']
        
        if exp_id is None:
            return
        
        exp = self.tracker.get_experiment(exp_id)
        
        if exp is None:
            return
        
        # Update metrics table
        metrics_df = pd.DataFrame([
            {'Metric': k, 'Value': f"{v:.4f}"} for k, v in exp['metrics'].items()
        ])
        
        self.metrics_table_html.value = metrics_df.to_html(index=False)
        
        # Plot training curves
        with self.training_plot_output:
            clear_output(wait=True)
            fig = self.analyzer.plot_training_curves(exp_id)
            if fig:
                display(fig)
    
    def _compare_experiments(self, button):
        """Compare selected experiments."""
        exp_ids = list(self.compare_selector.value)
        
        if len(exp_ids) < 2:
            with self.compare_output:
                clear_output(wait=True)
                print("Please select at least 2 experiments to compare")
            return
        
        comparison = self.tracker.compare_experiments(exp_ids)
        
        with self.compare_output:
            clear_output(wait=True)
            
            # Build comparison table
            rows = []
            for exp_data in comparison['comparison']['experiments']:
                row = {
                    'ID': exp_data['id'],
                    'Name': exp_data['name'][:40],
                }
                row.update(exp_data['metrics'])
                rows.append(row)
            
            df = pd.DataFrame(rows)
            display(HTML("<h4>Experiment Comparison</h4>"))
            display(df)
    
    def _refresh_campaigns(self, button):
        """Refresh campaign list."""
        experiments = self.tracker.get_all_experiments(status='completed')
        
        # Extract unique campaign names
        campaigns = set()
        for exp in experiments:
            if exp['campaign_name']:
                campaigns.add(exp['campaign_name'])
        
        if campaigns:
            self.campaign_selector.options = sorted(campaigns)
        else:
            self.campaign_selector.options = [('No campaigns found', None)]
    
    def _generate_report(self, button):
        """Generate analysis report for selected campaign."""
        campaign_name = self.campaign_selector.value
        
        if not campaign_name or campaign_name == 'No campaigns found':
            with self.analysis_output:
                clear_output(wait=True)
                print("Please select a campaign")
            return
        
        with self.analysis_output:
            clear_output(wait=True)
            
            print(f"Generating analysis for campaign: {campaign_name}")
            print("="*60)
            
            # Probe comparison
            print("\n1. Probe Comparison")
            probe_df = self.analyzer.compare_probes(campaign_name=campaign_name)
            if not probe_df.empty:
                display(probe_df)
                fig = self.analyzer.plot_probe_comparison(campaign_name=campaign_name)
                if fig:
                    display(fig)
            else:
                print("No data available")
            
            # Model comparison
            print("\n2. Model Comparison")
            model_df = self.analyzer.compare_models(campaign_name=campaign_name)
            if not model_df.empty:
                display(model_df)
                fig = self.analyzer.plot_model_comparison(campaign_name=campaign_name)
                if fig:
                    display(fig)
            else:
                print("No data available")
            
            # Sparsity analysis
            print("\n3. Sparsity Analysis")
            sparsity_df = self.analyzer.sparsity_analysis(campaign_name=campaign_name)
            if not sparsity_df.empty:
                display(sparsity_df.head(10))
                fig = self.analyzer.plot_sparsity_analysis(campaign_name=campaign_name)
                if fig:
                    display(fig)
            else:
                print("No data available")
    
    def _apply_filters(self, button):
        """Apply filters and display database table."""
        experiments = self.tracker.get_all_experiments(status='completed')
        
        # Apply filters
        probe_filter = self.filter_probe_text.value.strip()
        model_filter = self.filter_model_text.value.strip()
        min_acc = self.filter_min_acc_text.value
        
        filtered = []
        for exp in experiments:
            if probe_filter and probe_filter.lower() not in exp['probe_type'].lower():
                continue
            if model_filter and model_filter.lower() not in exp['model_type'].lower():
                continue
            if min_acc > 0:
                acc = exp['metrics'].get('top_1_accuracy', 0.0)
                if acc < min_acc:
                    continue
            filtered.append(exp)
        
        # Sort
        sort_by = self.sort_dropdown.value
        if sort_by == 'timestamp':
            filtered.sort(key=lambda e: e['timestamp'], reverse=True)
        elif sort_by == 'top_1_accuracy':
            filtered.sort(key=lambda e: e['metrics'].get('top_1_accuracy', 0.0), reverse=True)
        elif sort_by == 'training_time_seconds':
            filtered.sort(key=lambda e: e['training_time_seconds'])
        
        # Build table
        rows = []
        for exp in filtered[:50]:  # Limit to 50 rows
            rows.append({
                'ID': exp['id'],
                'Name': exp['name'][:40],
                'Probe': exp['probe_type'],
                'Model': exp['model_type'],
                'M': exp['M'],
                'K': exp['K'],
                'Accuracy': f"{exp['metrics'].get('top_1_accuracy', 0.0):.3f}",
                'Time': f"{exp['training_time_seconds']:.1f}s",
                'Timestamp': exp['timestamp'][:16],
            })
        
        with self.db_table_output:
            clear_output(wait=True)
            if rows:
                df = pd.DataFrame(rows)
                display(HTML(f"<p>Showing {len(rows)} of {len(filtered)} experiments (top 50)</p>"))
                display(df)
            else:
                print("No experiments match the filters")
    
    def _export_csv(self, button):
        """Export experiments to CSV."""
        try:
            output_path = f"outputs/experiments_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.tracker.export_to_csv(output_path)
            
            with self.db_table_output:
                print(f"\n✅ Exported to {output_path}")
        except Exception as e:
            with self.db_table_output:
                print(f"\n❌ Export failed: {e}")
    
    def _export_json(self, button):
        """Export experiments to JSON."""
        import json
        
        try:
            experiments = self.tracker.get_all_experiments()
            
            output_path = f"outputs/experiments_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(output_path, 'w') as f:
                json.dump(experiments, f, indent=2)
            
            with self.db_table_output:
                print(f"\n✅ Exported to {output_path}")
        except Exception as e:
            with self.db_table_output:
                print(f"\n❌ Export failed: {e}")
    
    def _update_db_stats(self):
        """Update database statistics."""
        experiments = self.tracker.get_all_experiments()
        
        if not experiments:
            self.db_stats_html.value = "<i>No experiments in database</i>"
            return
        
        total = len(experiments)
        completed = len([e for e in experiments if e['status'] == 'completed'])
        failed = len([e for e in experiments if e['status'] == 'failed'])
        
        # Find best
        best = self.tracker.get_best_experiment()
        
        stats_html = f"""
        <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
            <h4>Database Statistics</h4>
            <ul>
                <li><b>Total Experiments:</b> {total}</li>
                <li><b>Completed:</b> {completed}</li>
                <li><b>Failed:</b> {failed}</li>
                {f"<li><b>Best Result:</b> {best['name']} ({best['metrics'].get('top_1_accuracy', 0.0):.3f})</li>" if best else ""}
            </ul>
        </div>
        """
        
        self.db_stats_html.value = stats_html
