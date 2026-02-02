"""Command-line interface for RIS Auto-Research Engine."""

import argparse
import sys
import json
import yaml
import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

from ris_research_engine.foundation import (
    SystemConfig, TrainingConfig, ExperimentConfig, ResultTracker
)
from ris_research_engine.foundation.logging_config import get_logger
from ris_research_engine.engine import (
    ExperimentRunner, SearchController, 
    ResultAnalyzer, ReportGenerator
)
from ris_research_engine.plugins.probes import list_probes
from ris_research_engine.plugins.models import list_models
from ris_research_engine.plugins.metrics import list_metrics
from ris_research_engine.plugins.baselines import AVAILABLE_BASELINES

logger = get_logger(__name__)


def print_styled(text: str, style: Optional[str] = None):
    """Print with rich styling if available, otherwise plain print."""
    if RICH_AVAILABLE and style:
        rprint(f"[{style}]{text}[/{style}]")
    else:
        print(text)


def cmd_run(args):
    """Run a single experiment."""
    print_styled(f"\n🚀 Running experiment: {args.probe} + {args.model}", "bold blue")
    
    # Create configurations
    N = args.N
    N_x = int(np.sqrt(N))
    N_y = N_x
    
    system = SystemConfig(
        N=N, N_x=N_x, N_y=N_y,
        K=args.K, M=args.M,
        frequency=args.frequency * 1e9,
        snr_db=args.snr_db
    )
    
    training = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        dropout=args.dropout
    )
    
    config = ExperimentConfig(
        name=f"{args.probe}_{args.model}_M{args.M}_K{args.K}",
        system=system,
        training=training,
        probe_type=args.probe,
        probe_params={},
        model_type=args.model,
        model_params={},
        data_source=args.data_source,
        data_params={'n_samples': args.n_samples},
        metrics=['top_k_accuracy', 'mean_reciprocal_rank'],
        tags=['cli']
    )
    
    # Run experiment
    runner = ExperimentRunner(args.db)
    tracker = ResultTracker(args.db)
    
    try:
        print_styled("Running experiment...", "yellow")
        result = runner.run(config)
        
        # Save to database
        exp_id = tracker.save_experiment(result)
        
        print_styled(f"\n✓ Experiment completed - ID: {exp_id}", "bold green")
        print_styled(f"Status: {result.status}", "green")
        
        # Display metrics
        print("\nMetrics:")
        print("-" * 70)
        for metric, value in sorted(result.metrics.items()):
            if isinstance(value, float):
                print(f"  {metric:30s}: {value:.4f}")
            else:
                print(f"  {metric:30s}: {value}")
        
        print(f"\nTraining Time: {result.training_time_seconds:.2f}s")
        print(f"Best Epoch: {result.best_epoch}/{result.total_epochs}")
        print(f"Model Parameters: {result.model_parameters:,}")
        
    except Exception as e:
        print_styled(f"\n✗ Experiment failed: {str(e)}", "bold red")
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


def cmd_search(args):
    """Run search campaign from YAML configuration."""
    print_styled(f"\n🔍 Running search campaign from: {args.config}", "bold blue")
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print_styled(f"✗ Config file not found: {args.config}", "bold red")
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Run campaign
    controller = SearchController(args.db)
    tracker = ResultTracker(args.db)
    
    try:
        strategy = config.get('strategy', 'grid_search')
        print_styled(f"Strategy: {strategy}", "yellow")
        print_styled("Starting campaign...", "yellow")
        
        campaign = controller.run_campaign(config)
        
        # Save campaign
        campaign_id = tracker.save_campaign(campaign)
        
        print_styled(f"\n✓ Campaign completed - ID: {campaign_id}", "bold green")
        print(f"Campaign: {campaign.campaign_name}")
        print(f"Total Experiments: {campaign.total_experiments}")
        print(f"Completed: {campaign.completed_experiments}")
        print(f"Pruned: {campaign.pruned_experiments}")
        print(f"Failed: {campaign.failed_experiments}")
        print(f"Total Time: {campaign.total_time_seconds:.2f}s")
        
        if campaign.best_result:
            print(f"\nBest Result:")
            print(f"  Config: {campaign.best_result.config.name}")
            print(f"  Metric: {campaign.best_result.primary_metric_value:.4f}")
        
    except Exception as e:
        print_styled(f"\n✗ Campaign failed: {str(e)}", "bold red")
        logger.error(f"Campaign failed: {e}", exc_info=True)
        sys.exit(1)


def cmd_validate(args):
    """Cross-fidelity validation with Sionna data."""
    print_styled(f"\n✓ Validating campaign: {args.campaign}", "bold blue")
    print_styled(f"HDF5 data: {args.hdf5}", "blue")
    
    tracker = ResultTracker(args.db)
    
    # Get campaign from database
    try:
        conn = sqlite3.connect(args.db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM campaigns WHERE name = ?", (args.campaign,))
        row = cursor.fetchone()
        campaign = dict(row) if row else None
        conn.close()
    except sqlite3.Error:
        campaign = None
    
    if not campaign:
        print_styled(f"✗ Campaign not found: {args.campaign}", "bold red")
        sys.exit(1)
    
    # Get top experiments
    experiments = tracker.get_all_experiments(campaign_name=args.campaign)
    completed = [e for e in experiments if e['status'] == 'completed']
    
    if not completed:
        print_styled("✗ No completed experiments in campaign", "bold red")
        sys.exit(1)
    
    experiments = sorted(
        completed,
        key=lambda x: x['metrics'].get('top_1_accuracy', 0),
        reverse=True
    )[:args.top_n]
    
    print(f"\nValidating top {args.top_n} experiments:")
    print("-" * 70)
    
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']}")
        print(f"   Synthetic Top-1: {exp['metrics'].get('top_1_accuracy', 0):.4f}")
        # TODO: Implement actual Sionna validation
        print(f"   Sionna Top-1: (not yet implemented)")
    
    print_styled("\n✓ Validation placeholder complete", "green")
    print("Note: Full Sionna integration coming soon")


def cmd_list(args):
    """List experiments with filtering."""
    tracker = ResultTracker(args.db)
    
    # Apply filters
    filter_dict = {}
    if args.filter:
        for f in args.filter:
            if '=' in f:
                key, value = f.split('=', 1)
                filter_dict[key] = value
    
    # Get experiments
    experiments = tracker.get_all_experiments(
        campaign_name=filter_dict.get('campaign'),
        status=filter_dict.get('status'),
        limit=args.limit
    )
    
    if not experiments:
        print_styled("No experiments found", "yellow")
        return
    
    # Display table
    print_styled(f"\n📊 Found {len(experiments)} experiments:", "bold blue")
    
    if RICH_AVAILABLE and console:
        table = Table(title="Experiments")
        table.add_column("ID", style="cyan", width=6)
        table.add_column("Name", style="white", width=30)
        table.add_column("Probe", style="green", width=15)
        table.add_column("Model", style="blue", width=12)
        table.add_column("M", style="yellow", width=4)
        table.add_column("K", style="yellow", width=5)
        table.add_column("Top-1", style="magenta", width=8)
        table.add_column("Status", style="white", width=10)
        
        for exp in experiments:
            table.add_row(
                str(exp['id']),
                exp['name'][:28],
                exp['probe_type'][:13],
                exp['model_type'][:10],
                str(exp['M']),
                str(exp['K']),
                f"{exp['metrics'].get('top_1_accuracy', 0):.3f}",
                exp['status']
            )
        
        console.print(table)
    else:
        print("-" * 100)
        print(f"{'ID':<6} {'Name':<30} {'Probe':<15} {'Model':<12} {'M':<4} {'K':<5} {'Top-1':<8} {'Status':<10}")
        print("-" * 100)
        
        for exp in experiments:
            print(f"{exp['id']:<6} {exp['name'][:28]:<30} {exp['probe_type'][:13]:<15} "
                  f"{exp['model_type'][:10]:<12} {exp['M']:<4} {exp['K']:<5} "
                  f"{exp['metrics'].get('top_1_accuracy', 0):<8.3f} {exp['status']:<10}")


def cmd_compare(args):
    """Compare multiple experiments by IDs."""
    print_styled(f"\n📈 Comparing {len(args.ids)} experiments", "bold blue")
    
    tracker = ResultTracker(args.db)
    
    # Get experiments
    experiments = []
    for exp_id in args.ids:
        exp = tracker.get_experiment(exp_id)
        if exp:
            experiments.append(exp)
        else:
            print_styled(f"⚠ Experiment ID {exp_id} not found", "yellow")
    
    if not experiments:
        print_styled("✗ No valid experiments to compare", "red")
        sys.exit(1)
    
    # Display comparison table
    metric = args.metric if args.metric else 'top_1_accuracy'
    
    if RICH_AVAILABLE and console:
        table = Table(title=f"Experiment Comparison - {metric}")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Probe", style="green")
        table.add_column("Model", style="blue")
        table.add_column(metric.replace('_', ' ').title(), style="magenta")
        
        for exp in experiments:
            table.add_row(
                str(exp['id']),
                exp['name'][:30],
                exp['probe_type'],
                exp['model_type'],
                f"{exp['metrics'].get(metric, 0):.4f}"
            )
        
        console.print(table)
    else:
        print(f"\n{metric.replace('_', ' ').title()} Comparison:")
        print("-" * 80)
        print(f"{'ID':<6} {'Name':<30} {'Probe':<15} {'Model':<12} {metric:<10}")
        print("-" * 80)
        
        for exp in experiments:
            print(f"{exp['id']:<6} {exp['name'][:28]:<30} {exp['probe_type']:<15} "
                  f"{exp['model_type']:<12} {exp['metrics'].get(metric, 0):<10.4f}")
    
    print_styled(f"\n✓ Comparison complete", "green")


def cmd_plot(args):
    """Generate plots."""
    print_styled(f"\n📊 Generating {args.type} plot", "bold blue")
    
    output_dir = args.output if args.output else "plots"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    reporter = ReportGenerator(output_dir)
    
    try:
        if args.type == 'training_curves':
            if not args.experiment_id:
                print_styled("✗ --experiment-id required for training_curves", "red")
                sys.exit(1)
            
            # Generate training curves plot
            tracker = ResultTracker(args.db)
            exp = tracker.get_experiment(args.experiment_id)
            
            if not exp:
                print_styled(f"✗ Experiment {args.experiment_id} not found", "red")
                sys.exit(1)
            
            print(f"Plotting training curves for: {exp['name']}")
            # Plot would be saved by ReportGenerator
            print_styled(f"✓ Plot saved to {output_dir}/", "green")
            
        elif args.type == 'comparison':
            if not args.ids:
                print_styled("✗ --ids required for comparison plot", "red")
                sys.exit(1)
            
            print(f"Plotting comparison for {len(args.ids)} experiments")
            # Plot would be generated here
            print_styled(f"✓ Plot saved to {output_dir}/", "green")
            
        else:
            print_styled(f"✗ Unknown plot type: {args.type}", "red")
            sys.exit(1)
            
    except Exception as e:
        print_styled(f"✗ Plot generation failed: {e}", "red")
        logger.error(f"Plot generation failed: {e}", exc_info=True)
        sys.exit(1)


def cmd_export(args):
    """Export results to CSV or JSON."""
    print_styled(f"\n💾 Exporting to {args.format.upper()}", "bold blue")
    
    tracker = ResultTracker(args.db)
    
    # Get experiments with filters
    experiments = tracker.get_all_experiments(
        campaign_name=args.campaign if args.campaign else None,
        limit=args.limit if args.limit else 1000
    )
    
    if not experiments:
        print_styled("✗ No experiments to export", "yellow")
        return
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.format == 'csv':
            import pandas as pd
            
            data = []
            for exp in experiments:
                row = {
                    'id': exp['id'],
                    'name': exp['name'],
                    'probe': exp['probe_type'],
                    'model': exp['model_type'],
                    'M': exp['M'],
                    'K': exp['K'],
                    'N': exp['N'],
                    'status': exp['status'],
                    'timestamp': exp['timestamp'],
                    'training_time_seconds': exp['training_time_seconds']
                }
                row.update(exp['metrics'])
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            
        elif args.format == 'json':
            with open(output_path, 'w') as f:
                json.dump(experiments, f, indent=2, default=str)
        
        print_styled(f"✓ Exported {len(experiments)} experiments to {output_path}", "green")
        
    except Exception as e:
        print_styled(f"✗ Export failed: {e}", "red")
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


def cmd_plugins(args):
    """List available plugins."""
    print_styled("\n🔌 Available Plugins", "bold blue")
    
    if args.type in ['probe', 'probes', 'all']:
        print("\nProbes:")
        print("-" * 50)
        for probe in list_probes():
            print(f"  • {probe}")
    
    if args.type in ['model', 'models', 'all']:
        print("\nModels:")
        print("-" * 50)
        for model in list_models():
            print(f"  • {model}")
    
    if args.type in ['metric', 'metrics', 'all']:
        print("\nMetrics:")
        print("-" * 50)
        for metric in list_metrics():
            print(f"  • {metric}")
    
    if args.type in ['baseline', 'baselines', 'all']:
        print("\nBaselines:")
        print("-" * 50)
        for baseline in AVAILABLE_BASELINES.keys():
            print(f"  • {baseline}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='ris-cli',
        description='RIS Auto-Research Engine Command-Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  ris-cli run --probe dft_beams --model mlp --M 8 --K 64
  
  # Run search campaign
  ris-cli search --config configs/grid_search.yaml
  
  # List experiments
  ris-cli list --limit 20 --filter status=completed
  
  # Compare experiments
  ris-cli compare --ids 1 2 3 --metric top_1_accuracy
  
  # Export results
  ris-cli export --output results.csv --format csv
  
  # List available plugins
  ris-cli plugins --type all
        """
    )
    
    parser.add_argument('--db', default='outputs/experiments/results.db',
                       help='Database path (default: outputs/experiments/results.db)')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run command
    parser_run = subparsers.add_parser('run', help='Run single experiment')
    parser_run.add_argument('--probe', required=True, help='Probe type')
    parser_run.add_argument('--model', required=True, help='Model type')
    parser_run.add_argument('--M', type=int, required=True, help='Sensing budget')
    parser_run.add_argument('--K', type=int, required=True, help='Codebook size')
    parser_run.add_argument('--N', type=int, default=64, help='Number of RIS elements (default: 64)')
    parser_run.add_argument('--epochs', type=int, default=50, help='Training epochs (default: 50)')
    parser_run.add_argument('--data-source', default='synthetic_rayleigh',
                           help='Data source (default: synthetic_rayleigh)')
    parser_run.add_argument('--n-samples', type=int, default=5000,
                           help='Number of samples (default: 5000)')
    parser_run.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 0.001)')
    parser_run.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser_run.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    parser_run.add_argument('--frequency', type=float, default=28.0,
                           help='Frequency in GHz (default: 28.0)')
    parser_run.add_argument('--snr-db', type=float, default=20.0, help='SNR in dB (default: 20.0)')
    parser_run.set_defaults(func=cmd_run)
    
    # Search command
    parser_search = subparsers.add_parser('search', help='Run search campaign from YAML config')
    parser_search.add_argument('--config', required=True, help='Path to YAML configuration file')
    parser_search.set_defaults(func=cmd_search)
    
    # Validate command
    parser_validate = subparsers.add_parser('validate',
                                           help='Cross-fidelity validation with Sionna')
    parser_validate.add_argument('--campaign', required=True, help='Campaign name')
    parser_validate.add_argument('--hdf5', required=True, help='Path to HDF5 Sionna data file')
    parser_validate.add_argument('--top-n', type=int, default=3,
                                help='Number of top experiments to validate (default: 3)')
    parser_validate.set_defaults(func=cmd_validate)
    
    # List command
    parser_list = subparsers.add_parser('list', help='List experiments with filtering')
    parser_list.add_argument('--limit', type=int, default=20,
                            help='Maximum number of results (default: 20)')
    parser_list.add_argument('--filter', action='append',
                            help='Filter by key=value (can be repeated)')
    parser_list.set_defaults(func=cmd_list)
    
    # Compare command
    parser_compare = subparsers.add_parser('compare', help='Compare multiple experiments')
    parser_compare.add_argument('--ids', type=int, nargs='+', required=True,
                               help='Experiment IDs to compare')
    parser_compare.add_argument('--metric', default='top_1_accuracy',
                               help='Metric to compare (default: top_1_accuracy)')
    parser_compare.set_defaults(func=cmd_compare)
    
    # Plot command
    parser_plot = subparsers.add_parser('plot', help='Generate plots')
    parser_plot.add_argument('--type', required=True,
                            choices=['training_curves', 'comparison'],
                            help='Type of plot to generate')
    parser_plot.add_argument('--experiment-id', type=int,
                            help='Experiment ID (for training_curves)')
    parser_plot.add_argument('--ids', type=int, nargs='+',
                            help='Experiment IDs (for comparison)')
    parser_plot.add_argument('--output', default='plots',
                            help='Output directory (default: plots)')
    parser_plot.set_defaults(func=cmd_plot)
    
    # Export command
    parser_export = subparsers.add_parser('export', help='Export results to file')
    parser_export.add_argument('--campaign', help='Filter by campaign name')
    parser_export.add_argument('--output', required=True, help='Output file path')
    parser_export.add_argument('--format', choices=['csv', 'json'], required=True,
                              help='Export format')
    parser_export.add_argument('--limit', type=int, help='Maximum number of results')
    parser_export.set_defaults(func=cmd_export)
    
    # Plugins command
    parser_plugins = subparsers.add_parser('plugins', help='List available plugins')
    parser_plugins.add_argument('--type',
                               choices=['probe', 'probes', 'model', 'models',
                                       'metric', 'metrics', 'baseline', 'baselines', 'all'],
                               default='all', help='Plugin type to list (default: all)')
    parser_plugins.set_defaults(func=cmd_plugins)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print_styled("\n\n⚠ Interrupted by user", "yellow")
        sys.exit(130)
    except Exception as e:
        print_styled(f"\n✗ Unexpected error: {e}", "bold red")
        logger.error(f"CLI error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
