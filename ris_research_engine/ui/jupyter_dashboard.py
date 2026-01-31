"""Interactive Jupyter dashboard using ipywidgets."""

import logging
from typing import Optional
import pandas as pd

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False
    logging.warning("ipywidgets not available, dashboard disabled")

from ris_research_engine.ui.jupyter_minimal import RISEngine

logger = logging.getLogger(__name__)


class RISDashboard:
    """Interactive dashboard for RIS research engine."""
    
    def __init__(self, db_path: str = "results.db", output_dir: str = "outputs"):
        """Initialize dashboard.
        
        Args:
            db_path: Path to results database
            output_dir: Directory for outputs
        """
        if not IPYWIDGETS_AVAILABLE:
            raise ImportError("ipywidgets is required for dashboard functionality")
        
        self.engine = RISEngine(db_path, output_dir)
        self.current_results = []
        
        # Create UI components
        self._create_ui()
    
    def _create_ui(self):
        """Create dashboard UI."""
        # Create tabs
        self.tabs = widgets.Tab()
        
        # Tab 1: Configure
        self.tab_configure = self._create_configure_tab()
        
        # Tab 2: Run
        self.tab_run = self._create_run_tab()
        
        # Tab 3: Results
        self.tab_results = self._create_results_tab()
        
        # Tab 4: Analysis
        self.tab_analysis = self._create_analysis_tab()
        
        # Tab 5: Database
        self.tab_database = self._create_database_tab()
        
        # Set tabs
        self.tabs.children = [
            self.tab_configure,
            self.tab_run,
            self.tab_results,
            self.tab_analysis,
            self.tab_database
        ]
        self.tabs.set_title(0, 'Configure')
        self.tabs.set_title(1, 'Run')
        self.tabs.set_title(2, 'Results')
        self.tabs.set_title(3, 'Analysis')
        self.tabs.set_title(4, 'Database')
    
    def _create_configure_tab(self):
        """Create configuration tab."""
        # System parameters
        n_widget = widgets.IntText(value=64, description='N (elements):')
        k_widget = widgets.IntText(value=64, description='K (codebook):')
        m_widget = widgets.IntText(value=8, description='M (probes):')
        snr_widget = widgets.FloatText(value=20.0, description='SNR (dB):')
        
        # Probe selection
        probe_widget = widgets.Dropdown(
            options=['random_uniform', 'hadamard', 'sobol', 'dft_beams'],
            description='Probe:',
            value='random_uniform'
        )
        
        # Model selection
        model_widget = widgets.Dropdown(
            options=['mlp', 'residual_mlp', 'cnn_1d', 'transformer'],
            description='Model:',
            value='mlp'
        )
        
        # Training parameters
        epochs_widget = widgets.IntText(value=100, description='Epochs:')
        lr_widget = widgets.FloatText(value=1e-3, description='Learning rate:')
        batch_widget = widgets.IntText(value=64, description='Batch size:')
        
        # Store widgets
        self.config_widgets = {
            'N': n_widget,
            'K': k_widget,
            'M': m_widget,
            'snr_db': snr_widget,
            'probe': probe_widget,
            'model': model_widget,
            'epochs': epochs_widget,
            'learning_rate': lr_widget,
            'batch_size': batch_widget,
        }
        
        return widgets.VBox([
            widgets.HTML("<h3>System Parameters</h3>"),
            n_widget, k_widget, m_widget, snr_widget,
            widgets.HTML("<h3>Method Selection</h3>"),
            probe_widget, model_widget,
            widgets.HTML("<h3>Training Parameters</h3>"),
            epochs_widget, lr_widget, batch_widget,
        ])
    
    def _create_run_tab(self):
        """Create run tab."""
        # Output area
        self.run_output = widgets.Output()
        
        # Run button
        run_button = widgets.Button(
            description='Run Experiment',
            button_style='success',
            icon='play'
        )
        run_button.on_click(self._on_run_clicked)
        
        # Progress bar
        self.progress_bar = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            description='Progress:',
            bar_style='info'
        )
        
        return widgets.VBox([
            widgets.HTML("<h3>Run Experiment</h3>"),
            run_button,
            self.progress_bar,
            self.run_output
        ])
    
    def _create_results_tab(self):
        """Create results tab."""
        # Results output
        self.results_output = widgets.Output()
        
        # Refresh button
        refresh_button = widgets.Button(
            description='Refresh Results',
            icon='refresh'
        )
        refresh_button.on_click(self._on_refresh_results)
        
        return widgets.VBox([
            widgets.HTML("<h3>Recent Results</h3>"),
            refresh_button,
            self.results_output
        ])
    
    def _create_analysis_tab(self):
        """Create analysis tab."""
        # Analysis output
        self.analysis_output = widgets.Output()
        
        # Analysis type selector
        analysis_type = widgets.Dropdown(
            options=['Probe Comparison', 'Model Comparison', 'Sparsity Analysis'],
            description='Analysis:',
            value='Probe Comparison'
        )
        
        # Run analysis button
        analyze_button = widgets.Button(
            description='Run Analysis',
            button_style='info',
            icon='bar-chart'
        )
        analyze_button.on_click(lambda b: self._on_analyze_clicked(analysis_type.value))
        
        self.analysis_type_widget = analysis_type
        
        return widgets.VBox([
            widgets.HTML("<h3>Analysis</h3>"),
            analysis_type,
            analyze_button,
            self.analysis_output
        ])
    
    def _create_database_tab(self):
        """Create database tab."""
        # Database output
        self.database_output = widgets.Output()
        
        # Campaign filter
        campaign_widget = widgets.Text(
            description='Campaign:',
            placeholder='Leave empty for all'
        )
        
        # Load button
        load_button = widgets.Button(
            description='Load Experiments',
            icon='database'
        )
        load_button.on_click(lambda b: self._on_load_database(campaign_widget.value or None))
        
        # Export button
        export_button = widgets.Button(
            description='Export to CSV',
            button_style='warning',
            icon='download'
        )
        export_button.on_click(self._on_export_clicked)
        
        return widgets.VBox([
            widgets.HTML("<h3>Database</h3>"),
            campaign_widget,
            widgets.HBox([load_button, export_button]),
            self.database_output
        ])
    
    def _on_run_clicked(self, button):
        """Handle run button click."""
        with self.run_output:
            clear_output()
            print("Starting experiment...")
            
            try:
                # Get configuration
                config = {k: w.value for k, w in self.config_widgets.items()}
                
                # Run experiment
                result = self.engine.run(
                    probe=config['probe'],
                    model=config['model'],
                    M=config['M'],
                    K=config['K'],
                    N=config['N'],
                    snr_db=config['snr_db'],
                    epochs=config['epochs'],
                    learning_rate=config['learning_rate'],
                    batch_size=config['batch_size'],
                )
                
                # Display results
                self.engine.show(result)
                self.current_results.append(result)
                
                print("\n✓ Experiment completed!")
                
            except Exception as e:
                print(f"✗ Error: {e}")
    
    def _on_refresh_results(self, button):
        """Handle refresh results button click."""
        with self.results_output:
            clear_output()
            
            df = self.engine.show_history(limit=20)
            
            if not df.empty:
                display(df)
            else:
                print("No results found")
    
    def _on_analyze_clicked(self, analysis_type):
        """Handle analyze button click."""
        with self.analysis_output:
            clear_output()
            
            print(f"Running {analysis_type}...")
            
            try:
                if analysis_type == 'Probe Comparison':
                    df = self.engine.analyzer.compare_probes()
                    display(df)
                    self.engine.reporter.probe_comparison_bar(df)
                    
                elif analysis_type == 'Model Comparison':
                    df = self.engine.analyzer.compare_models()
                    display(df)
                    self.engine.reporter.model_comparison_bar(df)
                    
                elif analysis_type == 'Sparsity Analysis':
                    df = self.engine.analyzer.sparsity_analysis()
                    display(df)
                    self.engine.reporter.sparsity_curve(df)
                
                print(f"\n✓ Analysis complete! Plots saved to {self.engine.reporter.output_dir}")
                
            except Exception as e:
                print(f"✗ Error: {e}")
    
    def _on_load_database(self, campaign_name):
        """Handle load database button click."""
        with self.database_output:
            clear_output()
            
            df = self.engine.show_history(campaign_name=campaign_name, limit=100)
            
            if not df.empty:
                display(df)
                print(f"\nLoaded {len(df)} experiments")
            else:
                print("No experiments found")
    
    def _on_export_clicked(self, button):
        """Handle export button click."""
        with self.database_output:
            results = self.engine.result_tracker.query(limit=1000)
            
            if results:
                # Simple export
                data = []
                for r in results:
                    data.append({
                        'name': r.config.name,
                        'probe': r.config.probe_type,
                        'model': r.config.model_type,
                        'accuracy': r.metrics.get('top_1_accuracy', 0.0),
                    })
                
                df = pd.DataFrame(data)
                df.to_csv('experiments_export.csv', index=False)
                print(f"✓ Exported {len(results)} experiments to experiments_export.csv")
            else:
                print("No experiments to export")
    
    def show(self):
        """Display the dashboard."""
        display(self.tabs)


def create_dashboard(db_path: str = "results.db", output_dir: str = "outputs"):
    """Create and display RIS dashboard.
    
    Args:
        db_path: Path to results database
        output_dir: Directory for outputs
        
    Returns:
        RISDashboard instance
    """
    dashboard = RISDashboard(db_path, output_dir)
    dashboard.show()
    return dashboard
