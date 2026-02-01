"""Command-line interface for RIS Auto-Research Engine."""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import List, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None
    Table = None

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


def print_output(text: str, style: Optional[str] = None):
    """Print with rich if available, otherwise plain print."""
    if RICH_AVAILABLE and style:
        rprint(f"[{style}]{text}[/{style}]")
    else:
        print(text)


def create_table(title: str, columns: List[str]) -> 'Table':
    """Create a rich table if available."""
    if RICH_AVAILABLE:
        table = Table(title=title)
        for col in columns:
            table.add_column(col)
        return table
    return None


def cmd_run(args):
    """Run a single experiment."""
    print_output(f"\n🚀 Running experiment: {args.probe} + {args.model}", "bold blue")
    
    # Create configurations
    import numpy as np
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
        max_epochs=args.epochs
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
    runner = ExperimentRunner()
    tracker = ResultTracker(args.db)
    
    try:
        print_output("Running experiment...", "yellow")
        result = runner.run(config)
        
        # Save to database
        exp_id = tracker.save_experiment(result)
        
        print_output(f"\n✓ Experiment completed - ID: {exp_id}", "bold green")
        print_output(f"Status: {result.status}", "green")
        
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
        print_output(f"\n✗ Experiment failed: {str(e)}", "bold red")
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


def cmd_search(args):
    """Run search campaign from YAML config."""
    print_output(f"\n🔍 Running search campaign from: {args.config}", "bold blue")
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print_output(f"✗ Config file not found: {args.config}", "bold red")
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Run campaign
    controller = SearchController(args.db)
    tracker = ResultTracker(args.db)
    
    try:
        print_output(f"Strategy: {config.get('strategy', 'grid')}", "yellow")
        print_output("Starting campaign...", "yellow")
        
        campaign = controller.run_campaign(
            search_space_config=config,
            strategy_name=config.get('strategy', 'grid')
        )
        
        # Save campaign
        campaign_id = tracker.save_campaign(campaign)
        
        print_output(f"\n✓ Campaign completed - ID: {campaign_id}", "bold green")
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
        print_output(f"\n✗ Campaign failed: {str(e)}", "bold red")
        logger.error(f"Campaign failed: {e}", exc_info=True)
        sys.exit(1)


def cmd_validate(args):
    """Cross-fidelity validation."""
    print_output(f"\n✓ Validating campaign: {args.campaign}", "bold blue")
    print_output(f"HDF5 data: {args.hdf5}", "blue")
    
    tracker = ResultTracker(args.db)
    
    # Get campaign - query database directly
    import sqlite3
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
        print_output(f"✗ Campaign not found: {args.campaign}", "bold red")
        sys.exit(1)
    
    # Get top experiments
    experiments = tracker.get_all_experiments(campaign_name=args.campaign)
    experiments = sorted(
        experiments,
        key=lambda x: x.get('primary_metric_value', 0),
        reverse=True
    )[:args.top_n]
    
    print(f"\nValidating top {args.top_n} experiments:")
    print("-" * 70)
    
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']}")
        print(f"   Synthetic Top-1: {exp['metrics'].get('top_1_accuracy', 0):.4f}")
        # TODO: Implement actual Sionna validation
        print(f"   Sionna Top-1: (not implemented)")
    
    print_output("\n✓ Validation placeholder complete", "green")


def cmd_list(args):
    """List experiments with filtering."""
    tracker = ResultTracker(args.db)
    
    # Get experiments
    experiments = tracker.get_all_experiments(
        campaign_name=args.campaign if args.campaign else None,
        status=args.status if args.status != 'all' else None,
        limit=args.limit
    )
    
    if not experiments:
        print_output("No experiments found", "yellow")
        return
    
    # Display table
    print_output(f"\n📊 Found {len(experiments)} experiments:", "bold blue")
    
    if RICH_AVAILABLE and console:
        table = Table(title="Experiments")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Probe", style="green")
        table.add_column("Model", style="blue")
        table.add_column("M", style="yellow")
        table.add_column("K", style="yellow")
        table.add_column("Top-1", style="magenta")
        table.add_column("Status", style="white")
        
        for exp in experiments:
            table.add_row(
                str(exp['id']),
                exp['name'][:30],
                exp['probe_type'],
                exp['model_type'],
                str(exp['M']),
                str(exp['K']),
                f"{exp['metrics'].get('top_1_accuracy', 0):.3f}",
                exp['status']
            )
        
        console.print(table)
    else:
        print("-" * 100)
        print(f"{'ID':<5} {'Name':<30} {'Probe':<15} {'Model':<12} {'M':<4} {'K':<5} {'Top-1':<8} {'Status':<10}")
        print("-" * 100)
        
        for exp in experiments:
            print(f"{exp['id']:<5} {exp['name'][:30]:<30} {exp['probe_type']:<15} "
                  f"{exp['model_type']:<12} {exp['M']:<4} {exp['K']:<5} "
                  f"{exp['metrics'].get('top_1_accuracy', 0):<8.3f} {exp['status']:<10}")


def cmd_compare(args):
    """Compare experiment IDs."""
    print_output(f"\n📈 Comparing {len(args.ids)} experiments", "bold blue")
    
    tracker = ResultTracker(args.db)
    analyzer = ResultAnalyzer(args.db)
    
    # Get comparison
    df = analyzer.compare_probes(args.ids)
    
    if df.empty:
        print_output("No data to compare", "yellow")
        return
    
    print("\n" + df.to_string(index=False))
    
    print_output(f"\n✓ Comparison complete", "green")


def cmd_plot(args):
    """Generate specific plot type."""
    print_output(f"\n📊 Generating {args.type} plot", "bold blue")
    
    reporter = ReportGenerator(args.output_dir or "plots")
    
    if args.type == 'probe_comparison':
        reporter.probe_comparison_bar(args.ids, metric=args.metric or 'top_1_accuracy')
        print_output(f"✓ Plot saved to {reporter.output_dir}", "green")
        
    elif args.type == 'training_curves':
        reporter.training_curves(args.ids[0] if args.ids else None)
        print_output(f"✓ Plot saved to {reporter.output_dir}", "green")
        
    else:
        print_output(f"✗ Unknown plot type: {args.type}", "red")
        sys.exit(1)


def cmd_export(args):
    """Export to CSV/JSON."""
    print_output(f"\n💾 Exporting to {args.format.upper()}", "bold blue")
    
    tracker = ResultTracker(args.db)
    
    # Get experiments
    experiments = tracker.get_all_experiments(
        campaign_name=args.campaign if args.campaign else None,
        limit=1000
    )
    
    output_path = Path(args.output)
    
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
                'status': exp['status'],
                'timestamp': exp['timestamp']
            }
            row.update(exp['metrics'])
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
    elif args.format == 'json':
        with open(output_path, 'w') as f:
            json.dump(experiments, f, indent=2)
    
    print_output(f"✓ Exported {len(experiments)} experiments to {output_path}", "green")


def cmd_plugins(args):
    """List available plugins."""
    print_output("\n🔌 Available Plugins", "bold blue")
    
    if args.type in ['probes', 'all']:
        print("\nProbes:")
        print("-" * 50)
        for probe in list_probes():
            print(f"  • {probe}")
    
    if args.type in ['models', 'all']:
        print("\nModels:")
        print("-" * 50)
        for model in list_models():
            print(f"  • {model}")
    
    if args.type in ['metrics', 'all']:
        print("\nMetrics:")
        print("-" * 50)
        for metric in list_metrics():
            print(f"  • {metric}")
    
    if args.type in ['baselines', 'all']:
        print("\nBaselines:")
        print("-" * 50)
        for baseline in AVAILABLE_BASELINES:
            print(f"  • {baseline}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='RIS Auto-Research Engine CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  ris-cli run --probe dft_beams --model mlp --M 8 --K 64
  
  # Run search campaign
  ris-cli search --config configs/grid_search.yaml
  
  # List experiments
  ris-cli list --limit 20
  
  # Compare experiments
  ris-cli compare --ids 1 2 3
  
  # Export results
  ris-cli export --campaign my_campaign --output results.csv --format csv
  
  # List available plugins
  ris-cli plugins --type all
        """
    )
    
    parser.add_argument('--db', default='results.db', 
                       help='Database path (default: results.db)')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run command
    parser_run = subparsers.add_parser('run', help='Run single experiment')
    parser_run.add_argument('--probe', required=True, help='Probe type')
    parser_run.add_argument('--model', required=True, help='Model type')
    parser_run.add_argument('--M', type=int, required=True, help='Sensing budget')
    parser_run.add_argument('--K', type=int, required=True, help='Codebook size')
    parser_run.add_argument('--N', type=int, default=64, help='Number of RIS elements')
    parser_run.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser_run.add_argument('--data-source', default='synthetic_rayleigh', 
                           help='Data source')
    parser_run.add_argument('--n-samples', type=int, default=1000, 
                           help='Number of samples')
    parser_run.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser_run.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser_run.add_argument('--frequency', type=float, default=28.0, 
                           help='Frequency in GHz')
    parser_run.add_argument('--snr-db', type=float, default=20.0, help='SNR in dB')
    parser_run.set_defaults(func=cmd_run)
    
    # Search command
    parser_search = subparsers.add_parser('search', help='Run search campaign')
    parser_search.add_argument('--config', required=True, help='Path to YAML config')
    parser_search.set_defaults(func=cmd_search)
    
    # Validate command
    parser_validate = subparsers.add_parser('validate', 
                                           help='Cross-fidelity validation')
    parser_validate.add_argument('--campaign', required=True, help='Campaign name')
    parser_validate.add_argument('--hdf5', required=True, help='HDF5 file path')
    parser_validate.add_argument('--top-n', type=int, default=3, 
                                help='Number of top experiments')
    parser_validate.set_defaults(func=cmd_validate)
    
    # List command
    parser_list = subparsers.add_parser('list', help='List experiments')
    parser_list.add_argument('--campaign', help='Filter by campaign')
    parser_list.add_argument('--status', choices=['all', 'completed', 'failed', 'pruned'],
                            default='all', help='Filter by status')
    parser_list.add_argument('--limit', type=int, default=20, help='Max results')
    parser_list.set_defaults(func=cmd_list)
    
    # Compare command
    parser_compare = subparsers.add_parser('compare', help='Compare experiments')
    parser_compare.add_argument('--ids', type=int, nargs='+', required=True,
                               help='Experiment IDs to compare')
    parser_compare.set_defaults(func=cmd_compare)
    
    # Plot command
    parser_plot = subparsers.add_parser('plot', help='Generate plots')
    parser_plot.add_argument('--type', required=True,
                            choices=['probe_comparison', 'training_curves'],
                            help='Plot type')
    parser_plot.add_argument('--ids', type=int, nargs='+', help='Experiment IDs')
    parser_plot.add_argument('--metric', default='top_1_accuracy', help='Metric to plot')
    parser_plot.add_argument('--output-dir', help='Output directory')
    parser_plot.set_defaults(func=cmd_plot)
    
    # Export command
    parser_export = subparsers.add_parser('export', help='Export results')
    parser_export.add_argument('--campaign', help='Campaign name')
    parser_export.add_argument('--output', required=True, help='Output file path')
    parser_export.add_argument('--format', choices=['csv', 'json'], required=True,
                              help='Export format')
    parser_export.set_defaults(func=cmd_export)
    
    # Plugins command
    parser_plugins = subparsers.add_parser('plugins', help='List available plugins')
    parser_plugins.add_argument('--type', 
                               choices=['probes', 'models', 'metrics', 'baselines', 'all'],
                               default='all', help='Plugin type')
    parser_plugins.set_defaults(func=cmd_plugins)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
