"""Command-line interface for RIS research engine."""

import argparse
import sys
import json
from pathlib import Path

from ris_research_engine.foundation import (
    SystemConfig, TrainingConfig, ExperimentConfig, ResultTracker
)
from ris_research_engine.engine import (
    ExperimentRunner, SearchController, ResultAnalyzer, ReportGenerator
)
from ris_research_engine.plugins.probes import list_probes
from ris_research_engine.plugins.models import list_models
from ris_research_engine.plugins.data_sources import list_data_sources
from ris_research_engine.plugins.metrics import list_metrics
from ris_research_engine.plugins.baselines import AVAILABLE_BASELINES


def cmd_run(args):
    """Run a single experiment."""
    # Compute N_x and N_y from N
    import math
    N_x = int(math.sqrt(args.N))
    N_y = N_x
    if N_x * N_y != args.N:
        # Try to find factors
        for n_x in range(int(math.sqrt(args.N)), 0, -1):
            if args.N % n_x == 0:
                N_x = n_x
                N_y = args.N // n_x
                break
    
    # Create configuration
    system = SystemConfig(
        N=args.N,
        N_x=N_x,
        N_y=N_y,
        K=args.K,
        M=args.M,
        snr_db=args.snr_db
    )
    
    training = TrainingConfig(
        max_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience
    )
    
    metrics = args.metrics.split(',') if args.metrics else ['top_1_accuracy', 'power_ratio']
    
    config = ExperimentConfig(
        name=args.name or f"{args.probe}_{args.model}_M{args.M}_K{args.K}",
        system=system,
        training=training,
        probe_type=args.probe,
        probe_params={},
        model_type=args.model,
        model_params={},
        data_source=args.data_source,
        data_params={'n_samples': args.n_samples},
        metrics=metrics,
        tags=[],
        notes=args.notes or ''
    )
    
    # Run experiment
    print(f"Running experiment: {config.name}")
    runner = ExperimentRunner(args.db)
    result = runner.run(config)
    
    # Display results
    print(f"\nStatus: {result.status}")
    
    if result.status == 'completed':
        print(f"\nMetrics:")
        for metric_name, metric_value in result.metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        print(f"\nTraining:")
        print(f"  Time: {result.training_time_seconds:.2f}s")
        print(f"  Epochs: {result.total_epochs}")
        print(f"  Best Epoch: {result.best_epoch}")
    else:
        print(f"Error: {result.error_message}")
    
    return 0 if result.status == 'completed' else 1


def cmd_search(args):
    """Run a search campaign from YAML configuration."""
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    print(f"Running search campaign from {config_path}")
    controller = SearchController(args.db)
    result = controller.run_from_yaml(str(config_path))
    
    # Display summary
    print(f"\nCampaign: {result.campaign_name}")
    print(f"Strategy: {result.search_strategy}")
    print(f"Total experiments: {result.total_experiments}")
    print(f"Completed: {result.completed_experiments}")
    print(f"Failed: {result.failed_experiments}")
    print(f"Time: {result.total_time_seconds:.2f}s")
    
    if result.best_result:
        print(f"\nBest result:")
        print(f"  Configuration: {result.best_result.config.name}")
        print(f"  {result.best_result.primary_metric_name}: {result.best_result.primary_metric_value:.4f}")
    
    return 0


def cmd_validate(args):
    """Run cross-fidelity validation."""
    print(f"Running cross-fidelity validation (top {args.top_n} experiments)")
    
    controller = SearchController(args.db)
    results = controller.run_cross_fidelity_validation(
        top_n=args.top_n,
        validation_fidelity=args.fidelity,
        validation_data_params={}
    )
    
    print(f"\nValidation completed for {len(results)} experiments")
    
    for i, result in enumerate(results):
        if result.status == 'completed':
            print(f"{i+1}. {result.config.name}: {result.primary_metric_name}={result.primary_metric_value:.4f}")
    
    return 0


def cmd_list(args):
    """List experiments from database."""
    tracker = ResultTracker(args.db)
    results = tracker.get_all_results()
    
    if not results:
        print("No experiments found.")
        return 0
    
    # Apply filters
    if args.status:
        results = [r for r in results if r.status == args.status]
    
    if args.probe:
        results = [r for r in results if r.config.probe_type == args.probe]
    
    if args.model:
        results = [r for r in results if r.config.model_type == args.model]
    
    # Sort by timestamp
    results.sort(key=lambda r: r.timestamp, reverse=True)
    
    # Limit results
    if args.limit:
        results = results[:args.limit]
    
    # Display results
    print(f"\nFound {len(results)} experiments\n")
    print(f"{'ID':<6} {'Name':<30} {'Status':<12} {'Probe':<15} {'Model':<10} {'Metric':<8}")
    print(f"{'-'*85}")
    
    for i, result in enumerate(results):
        name = result.config.name[:28] + '..' if len(result.config.name) > 30 else result.config.name
        primary_value = f"{result.primary_metric_value:.3f}" if result.status == 'completed' else "N/A"
        
        print(f"{i:<6} {name:<30} {result.status:<12} {result.config.probe_type:<15} "
              f"{result.config.model_type:<10} {primary_value:<8}")
    
    return 0


def cmd_compare(args):
    """Compare specific experiments."""
    tracker = ResultTracker(args.db)
    results = tracker.get_all_results()
    
    # Get experiments by IDs
    experiment_ids = [int(x) for x in args.ids.split(',')]
    selected_results = []
    
    for exp_id in experiment_ids:
        if exp_id < len(results):
            selected_results.append(results[exp_id])
        else:
            print(f"Warning: Experiment ID {exp_id} not found")
    
    if not selected_results:
        print("No valid experiments to compare")
        return 1
    
    # Display comparison
    print(f"\nComparing {len(selected_results)} experiments\n")
    
    metric = args.metric or 'top_1_accuracy'
    
    print(f"{'Name':<30} {'Probe':<15} {'Model':<10} {metric:<12}")
    print(f"{'-'*70}")
    
    for result in selected_results:
        name = result.config.name[:28] + '..' if len(result.config.name) > 30 else result.config.name
        value = f"{result.metrics.get(metric, 0.0):.4f}" if result.status == 'completed' else "N/A"
        
        print(f"{name:<30} {result.config.probe_type:<15} {result.config.model_type:<10} {value:<12}")
    
    return 0


def cmd_plot(args):
    """Generate a plot."""
    reporter = ReportGenerator(args.db)
    
    plot_type = args.type
    metric = args.metric or 'top_1_accuracy'
    
    if plot_type == 'probe_comparison':
        path = reporter.probe_comparison_bar(metric)
    elif plot_type == 'model_comparison':
        path = reporter.model_comparison_bar(metric)
    elif plot_type == 'sparsity':
        path = reporter.sparsity_curve(metric)
    elif plot_type == 'heatmap':
        path = reporter.heatmap_probe_model(metric)
    elif plot_type == 'pareto':
        metric_x = args.metric_x or 'training_time'
        metric_y = args.metric_y or 'top_1_accuracy'
        path = reporter.pareto_front(metric_x, metric_y)
    else:
        print(f"Unknown plot type: {plot_type}")
        print("Available types: probe_comparison, model_comparison, sparsity, heatmap, pareto")
        return 1
    
    if path:
        print(f"Plot saved to: {path}")
        return 0
    else:
        print("Failed to generate plot")
        return 1


def cmd_export(args):
    """Export experiment data."""
    analyzer = ResultAnalyzer(args.db)
    df = analyzer.get_results_dataframe({'status': 'completed'})
    
    if df.empty:
        print("No completed experiments to export")
        return 1
    
    output_path = Path(args.output)
    
    if args.format == 'csv':
        df.to_csv(output_path, index=False)
    elif args.format == 'json':
        df.to_json(output_path, orient='records', indent=2)
    else:
        print(f"Unknown format: {args.format}")
        return 1
    
    print(f"Exported {len(df)} experiments to {output_path}")
    return 0


def cmd_plugins(args):
    """List available plugins."""
    print("\nAvailable Plugins\n")
    print("=" * 60)
    
    print("\nProbes:")
    for probe in list_probes():
        print(f"  - {probe}")
    
    print("\nModels:")
    for model in list_models():
        print(f"  - {model}")
    
    print("\nData Sources:")
    for source in list_data_sources():
        print(f"  - {source}")
    
    print("\nMetrics:")
    for metric in list_metrics():
        print(f"  - {metric}")
    
    print("\nBaselines:")
    for baseline in AVAILABLE_BASELINES.keys():
        print(f"  - {baseline}")
    
    print("")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RIS Research Engine - Command Line Interface"
    )
    
    parser.add_argument(
        '--db',
        default='ris_results.db',
        help='Path to SQLite database (default: ris_results.db)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a single experiment')
    run_parser.add_argument('--probe', required=True, help='Probe type')
    run_parser.add_argument('--model', required=True, help='Model type')
    run_parser.add_argument('--N', type=int, default=64, help='Number of RIS elements')
    run_parser.add_argument('--K', type=int, default=64, help='Codebook size')
    run_parser.add_argument('--M', type=int, default=8, help='Sensing budget')
    run_parser.add_argument('--data-source', default='synthetic_rayleigh', help='Data source')
    run_parser.add_argument('--n-samples', type=int, default=1000, help='Number of training samples')
    run_parser.add_argument('--metrics', help='Comma-separated metric names')
    run_parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    run_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    run_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    run_parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    run_parser.add_argument('--snr-db', type=float, default=20.0, help='SNR in dB')
    run_parser.add_argument('--name', help='Experiment name')
    run_parser.add_argument('--notes', help='Experiment notes')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Run search campaign from YAML')
    search_parser.add_argument('--config', required=True, help='Path to YAML config file')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Run cross-fidelity validation')
    validate_parser.add_argument('--top-n', type=int, default=10, help='Number of top experiments to validate')
    validate_parser.add_argument('--fidelity', default='sionna', help='Target fidelity level')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--status', help='Filter by status')
    list_parser.add_argument('--probe', help='Filter by probe type')
    list_parser.add_argument('--model', help='Filter by model type')
    list_parser.add_argument('--limit', type=int, help='Limit number of results')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare experiments by IDs')
    compare_parser.add_argument('--ids', required=True, help='Comma-separated experiment IDs')
    compare_parser.add_argument('--metric', help='Metric to compare')
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Generate plot')
    plot_parser.add_argument('--type', required=True, 
                            help='Plot type (probe_comparison, model_comparison, sparsity, heatmap, pareto)')
    plot_parser.add_argument('--metric', help='Primary metric')
    plot_parser.add_argument('--metric-x', help='X-axis metric (for pareto)')
    plot_parser.add_argument('--metric-y', help='Y-axis metric (for pareto)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export experiment data')
    export_parser.add_argument('--output', required=True, help='Output file path')
    export_parser.add_argument('--format', choices=['csv', 'json'], default='csv', help='Output format')
    
    # Plugins command
    subparsers.add_parser('plugins', help='List available plugins')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'run':
        return cmd_run(args)
    elif args.command == 'search':
        return cmd_search(args)
    elif args.command == 'validate':
        return cmd_validate(args)
    elif args.command == 'list':
        return cmd_list(args)
    elif args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'plot':
        return cmd_plot(args)
    elif args.command == 'export':
        return cmd_export(args)
    elif args.command == 'plugins':
        return cmd_plugins(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
