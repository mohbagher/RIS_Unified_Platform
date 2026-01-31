"""Command-line interface for RIS research engine."""

import argparse
import sys
import logging
from pathlib import Path
import json
import pandas as pd

from ris_research_engine.foundation.storage import ResultTracker
from ris_research_engine.foundation.data_types import SystemConfig, TrainingConfig, ExperimentConfig
from ris_research_engine.engine.experiment_runner import ExperimentRunner
from ris_research_engine.engine.search_controller import SearchController
from ris_research_engine.engine.result_analyzer import ResultAnalyzer
from ris_research_engine.engine.report_generator import ReportGenerator
from ris_research_engine.plugins.probes import list_probes
from ris_research_engine.plugins.models import list_models
from ris_research_engine.plugins.metrics import list_metrics
from ris_research_engine.plugins.data_sources import list_data_sources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_run(args):
    """Run a single experiment."""
    logger.info("Running single experiment")
    
    # Create configs
    N_x = int(args.N ** 0.5)
    N_y = N_x
    
    system = SystemConfig(
        N=args.N,
        N_x=N_x,
        N_y=N_y,
        K=args.K,
        M=args.M,
        snr_db=args.snr_db,
    )
    
    training = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        random_seed=args.seed,
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
        metrics=args.metrics or ['top_1_accuracy'],
    )
    
    # Run experiment
    tracker = ResultTracker(args.db_path)
    runner = ExperimentRunner(tracker)
    
    result = runner.run(config)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Experiment: {result.config.name}")
    print(f"{'='*60}")
    print(f"Status: {result.status}")
    print(f"Training time: {result.training_time_seconds:.2f}s")
    print(f"\nTest Metrics:")
    for metric_name, value in result.metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    return 0


def cmd_search(args):
    """Run automated search campaign."""
    logger.info(f"Running search from config: {args.config}")
    
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    tracker = ResultTracker(args.db_path)
    controller = SearchController(tracker)
    
    # Run search
    result = controller.run_from_yaml(args.config)
    
    # Display summary
    print(f"\n{'='*60}")
    print(f"Search Campaign: {result.campaign_name}")
    print(f"{'='*60}")
    print(f"Strategy: {result.search_strategy}")
    print(f"Total experiments: {result.total_experiments}")
    print(f"Completed: {result.completed_experiments}")
    print(f"Pruned: {result.pruned_experiments}")
    print(f"Failed: {result.failed_experiments}")
    print(f"Total time: {result.total_time_seconds:.2f}s")
    
    if result.best_result:
        print(f"\nBest Result:")
        print(f"  Config: {result.best_result.config.name}")
        print(f"  Metric: {result.best_result.primary_metric_value:.4f}")
    
    return 0


def cmd_validate(args):
    """Run cross-fidelity validation."""
    logger.info(f"Running cross-fidelity validation")
    
    tracker = ResultTracker(args.db_path)
    controller = SearchController(tracker)
    
    df = controller.run_cross_fidelity_validation(
        args.campaign,
        args.hdf5_path,
        args.top_n
    )
    
    print(f"\nCross-Fidelity Validation Results:")
    print(df.to_string(index=False))
    
    # Save to CSV
    output_path = Path(args.output) if args.output else Path(f"validation_{args.campaign}.csv")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return 0


def cmd_list(args):
    """List experiments in database."""
    tracker = ResultTracker(args.db_path)
    
    results = tracker.query(
        campaign_name=args.campaign if hasattr(args, 'campaign') and args.campaign else None,
        status=args.status if hasattr(args, 'status') and args.status else None,
        limit=args.limit if hasattr(args, 'limit') else 50
    )
    
    if not results:
        print("No experiments found")
        return 0
    
    # Create summary table
    data = []
    for result in results:
        data.append({
            'Name': result.config.name,
            'Probe': result.config.probe_type,
            'Model': result.config.model_type,
            'M': result.config.system.M,
            'K': result.config.system.K,
            'Accuracy': f"{result.metrics.get('top_1_accuracy', 0.0):.4f}",
            'Status': result.status,
            'Time': f"{result.training_time_seconds:.1f}s",
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    return 0


def cmd_compare(args):
    """Compare experiments."""
    tracker = ResultTracker(args.db_path)
    analyzer = ResultAnalyzer(tracker)
    
    results = tracker.query(campaign_name=args.campaign) if args.campaign else tracker.query()
    
    if not results:
        print("No experiments found")
        return 0
    
    # Generate comparisons
    if args.type == 'probes':
        df = analyzer.compare_probes(results=results, metric=args.metric)
        print("\nProbe Comparison:")
    elif args.type == 'models':
        df = analyzer.compare_models(results=results, metric=args.metric)
        print("\nModel Comparison:")
    elif args.type == 'sparsity':
        df = analyzer.sparsity_analysis(results=results, metric=args.metric)
        print("\nSparsity Analysis:")
    else:
        print(f"Unknown comparison type: {args.type}")
        return 1
    
    print(df.to_string(index=False))
    
    return 0


def cmd_plot(args):
    """Generate plots."""
    tracker = ResultTracker(args.db_path)
    analyzer = ResultAnalyzer(tracker)
    reporter = ReportGenerator(args.output_dir)
    
    results = tracker.query(campaign_name=args.campaign) if args.campaign else tracker.query()
    
    if not results:
        print("No experiments found")
        return 0
    
    # Generate plots based on type
    if args.type == 'probes':
        df = analyzer.compare_probes(results=results)
        reporter.probe_comparison_bar(df)
    elif args.type == 'models':
        df = analyzer.compare_models(results=results)
        reporter.model_comparison_bar(df)
    elif args.type == 'sparsity':
        df = analyzer.sparsity_analysis(results=results)
        reporter.sparsity_curve(df)
    elif args.type == 'heatmap':
        reporter.heatmap_probe_model(results)
    elif args.type == 'pareto':
        reporter.pareto_front(results)
    elif args.type == 'distribution':
        reporter.ranking_distribution(results)
    elif args.type == 'all':
        # Generate all plots
        reporter.probe_comparison_bar(analyzer.compare_probes(results=results))
        reporter.model_comparison_bar(analyzer.compare_models(results=results))
        reporter.sparsity_curve(analyzer.sparsity_analysis(results=results))
        reporter.heatmap_probe_model(results)
        reporter.pareto_front(results)
        reporter.ranking_distribution(results)
    else:
        print(f"Unknown plot type: {args.type}")
        return 1
    
    print(f"Plots saved to: {reporter.output_dir}")
    
    return 0


def cmd_export(args):
    """Export results to CSV."""
    tracker = ResultTracker(args.db_path)
    
    results = tracker.query(campaign_name=args.campaign) if args.campaign else tracker.query()
    
    if not results:
        print("No experiments found")
        return 0
    
    # Export to CSV
    data = []
    for result in results:
        row = {
            'name': result.config.name,
            'timestamp': result.timestamp,
            'probe_type': result.config.probe_type,
            'model_type': result.config.model_type,
            'N': result.config.system.N,
            'K': result.config.system.K,
            'M': result.config.system.M,
            'status': result.status,
            'training_time': result.training_time_seconds,
            'model_params': result.model_parameters,
        }
        
        # Add metrics
        for metric_name, value in result.metrics.items():
            row[metric_name] = value
        
        data.append(row)
    
    df = pd.DataFrame(data)
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    
    print(f"Exported {len(results)} experiments to: {output_path}")
    
    return 0


def cmd_plugins(args):
    """List available plugins."""
    print("\n=== Available Plugins ===\n")
    
    print("Probes:")
    for probe in list_probes():
        print(f"  - {probe}")
    
    print("\nModels:")
    for model in list_models():
        print(f"  - {model}")
    
    print("\nMetrics:")
    for metric in list_metrics():
        print(f"  - {metric}")
    
    print("\nData Sources:")
    for ds in list_data_sources():
        print(f"  - {ds}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RIS Auto-Research Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument('--db-path', default='results.db', help='Path to results database')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a single experiment')
    run_parser.add_argument('--probe', required=True, help='Probe type')
    run_parser.add_argument('--model', required=True, help='Model type')
    run_parser.add_argument('--M', type=int, required=True, help='Sensing budget')
    run_parser.add_argument('--K', type=int, default=64, help='Codebook size')
    run_parser.add_argument('--N', type=int, default=64, help='Number of RIS elements')
    run_parser.add_argument('--data-source', default='synthetic_rayleigh', help='Data source')
    run_parser.add_argument('--n-samples', type=int, default=10000, help='Number of samples')
    run_parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    run_parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    run_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    run_parser.add_argument('--snr-db', type=float, default=20.0, help='SNR in dB')
    run_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    run_parser.add_argument('--metrics', nargs='+', help='Metrics to compute')
    run_parser.set_defaults(func=cmd_run)
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Run automated search')
    search_parser.add_argument('--config', required=True, help='Path to search config YAML')
    search_parser.set_defaults(func=cmd_search)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Cross-fidelity validation')
    validate_parser.add_argument('--campaign', required=True, help='Campaign name')
    validate_parser.add_argument('--hdf5-path', required=True, help='Path to Sionna HDF5 file')
    validate_parser.add_argument('--top-n', type=int, default=10, help='Number of top configs')
    validate_parser.add_argument('--output', help='Output CSV path')
    validate_parser.set_defaults(func=cmd_validate)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--campaign', help='Filter by campaign name')
    list_parser.add_argument('--status', help='Filter by status')
    list_parser.add_argument('--limit', type=int, default=50, help='Max results')
    list_parser.set_defaults(func=cmd_list)
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('--type', required=True, 
                               choices=['probes', 'models', 'sparsity'],
                               help='Comparison type')
    compare_parser.add_argument('--campaign', help='Filter by campaign')
    compare_parser.add_argument('--metric', default='top_1_accuracy', help='Metric to compare')
    compare_parser.set_defaults(func=cmd_compare)
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Generate plots')
    plot_parser.add_argument('--type', required=True,
                            choices=['probes', 'models', 'sparsity', 'heatmap', 'pareto', 'distribution', 'all'],
                            help='Plot type')
    plot_parser.add_argument('--campaign', help='Filter by campaign')
    plot_parser.add_argument('--output-dir', default='outputs', help='Output directory')
    plot_parser.set_defaults(func=cmd_plot)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export results to CSV')
    export_parser.add_argument('--campaign', help='Filter by campaign')
    export_parser.add_argument('--output', required=True, help='Output CSV path')
    export_parser.set_defaults(func=cmd_export)
    
    # Plugins command
    plugins_parser = subparsers.add_parser('plugins', help='List available plugins')
    plugins_parser.set_defaults(func=cmd_plugins)
    
    # Parse and execute
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
