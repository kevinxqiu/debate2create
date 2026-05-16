#!/usr/bin/env python3
"""
Plot debate performance metrics from Debate2Create runs.

This script generates two types of visualizations:
1. Best score at each debate round
2. Average score with 95% confidence interval at each round

Features:
- Automatically uses matplotlib for professional plots when available
- Falls back to text-based visualization when matplotlib is not installed
- Saves plots directly in the run directory by default
- Exports data as CSV files for further analysis

Usage:
    PYTHONPATH=src:. python scripts/plot_debate_performance.py \
        [--run_dir RUN_DIR] [--output_dir OUTPUT_DIR] [--use_matplotlib]
"""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import List, Tuple, Optional
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    import yaml
except ImportError:
    yaml = None

# Try to import matplotlib, but don't fail if not available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Will use text-based visualization only.")


def extract_scores_from_run(run_dir: Path) -> Tuple[List[float], List[float]]:
    """
    Extract scores from a debate run directory.

    Returns:
        Tuple of (best_scores_per_round, all_scores_per_round)
        - best_scores_per_round: List of best scores for each round
        - all_scores_per_round: List of lists, each containing all scores for that round
    """
    best_scores = []
    all_scores_per_round = []
    cumulative_best = float('-inf')

    # Find all round directories
    round_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith('round_')])

    for round_dir in round_dirs:
        round_scores = []

        # Look for training metrics files anywhere under the round directory (thesis_XX, synthesis_XX, etc.)
        metrics_files = list(round_dir.glob('**/train_metrics_*.json'))

        for metrics_file in metrics_files:
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                score = metrics.get("eval/episode_distance_from_origin")
                if score is None:
                    continue
                round_scores.append(score)

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not read {metrics_file}: {e}")
                continue

        if round_scores:
            round_best = max(round_scores)
            # Enforce non-decreasing best: carry forward prior best if higher (e.g., champion replay / previous champ)
            cumulative_best = max(cumulative_best, round_best)
            best_scores.append(cumulative_best)
            # Add all scores for this round
            all_scores_per_round.append(round_scores)
        else:
            print(f"Warning: No valid scores found in {round_dir}")

    return best_scores, all_scores_per_round


def _collect_scores_from_dir(parent: Path) -> List[float]:
    scores: List[float] = []
    metrics_files = list(parent.glob('train_metrics_*.json'))
    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            score = metrics.get("eval/episode_distance_from_origin")
            if score is None:
                continue
            score = float(score)
            scores.append(score)
        except Exception:
            continue
    return scores


def _load_skip_thesis_flag(run_dir: Path) -> bool:
    """Detect if the run used thesis-eval skipping (from saved config)."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists() or yaml is None:
        return False
    try:
        cfg = yaml.safe_load(cfg_path.read_text())
        return bool(cfg.get("debate", {}).get("skip_thesis_eval", False))
    except Exception:
        return False


def extract_thesis_synthesis_scores_from_run(run_dir: Path) -> Tuple[List[float], List[float], List[List[float]], List[List[float]]]:
    """Extract per-round best scores for Thesis and Synthesis.

    Returns:
        (thesis_best_per_round, synthesis_best_per_round, thesis_all_scores, synthesis_all_scores)
    """
    thesis_best: List[float] = []
    synthesis_best: List[float] = []
    thesis_all: List[List[float]] = []
    synthesis_all: List[List[float]] = []
    skip_thesis = _load_skip_thesis_flag(run_dir)

    round_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith('round_')])
    for round_dir in round_dirs:
        # Support multiple candidates: thesis_XX, synthesis_XX
        thesis_dirs = [] if skip_thesis else sorted([d for d in round_dir.iterdir() if d.is_dir() and d.name.startswith('thesis')])
        synth_dirs = sorted([d for d in round_dir.iterdir() if d.is_dir() and d.name.startswith('synthesis')])

        if not skip_thesis:
            t_scores_all: List[float] = []
            for td in thesis_dirs:
                t_scores_all.extend(_collect_scores_from_dir(td))

            if t_scores_all:
                thesis_all.append(t_scores_all)
                thesis_best.append(max(t_scores_all))
            else:
                print(f"Warning: No valid scores found in any thesis* dir under {round_dir}")
                thesis_all.append([])
                thesis_best.append(float('nan'))

        s_scores_all: List[float] = []
        for sd in synth_dirs:
            s_scores_all.extend(_collect_scores_from_dir(sd))
        if s_scores_all:
            synthesis_all.append(s_scores_all)
            synthesis_best.append(max(s_scores_all))
        else:
            print(f"Warning: No valid scores found in any synthesis* dir under {round_dir}")
            synthesis_all.append([])
            synthesis_best.append(float('nan'))

    if skip_thesis:
        thesis_best, thesis_all = [], []

    return thesis_best, synthesis_best, thesis_all, synthesis_all


def calculate_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate mean and confidence interval for a list of scores.

    Args:
        scores: List of scores
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    # Drop non-finite values (nan/inf) to keep statistics module happy
    finite_scores = [s for s in scores if isinstance(s, (int, float)) and math.isfinite(s)]
    if not finite_scores:
        return 0.0, 0.0, 0.0

    mean = statistics.mean(finite_scores)
    std = statistics.stdev(finite_scores) if len(finite_scores) > 1 else 0.0  # Sample standard deviation
    n = len(finite_scores)

    # Calculate confidence interval using t-distribution approximation
    # For simplicity, using normal distribution approximation (valid for large n)
    z_critical = 1.96 if confidence == 0.95 else 2.576  # 95% or 99% CI
    margin_error = z_critical * (std / math.sqrt(n))

    lower_bound = mean - margin_error
    upper_bound = mean + margin_error

    return mean, lower_bound, upper_bound


def save_best_scores(best_scores: List[float], output_dir: Path, run_name: str = ""):
    """Save best scores data and create a simple visualization."""
    if not best_scores:
        print("No best scores to save")
        return

    # Save as CSV
    csv_file = output_dir / f'best_scores{"_" + run_name if run_name else ""}.csv'
    with open(csv_file, 'w') as f:
        f.write("Round,Best_Score\n")
        for round_num, score in enumerate(best_scores):
            f.write(f"{round_num},{score:.3f}\n")

    # Create simple text visualization
    viz_file = output_dir / f'best_scores{"_" + run_name if run_name else ""}.txt'
    with open(viz_file, 'w') as f:
        f.write(f"Best Scores per Round{f' - {run_name}' if run_name else ''}\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Round':<8} {'Score':<12} {'Visualization'}\n")
        f.write("-" * 50 + "\n")

        max_score = max(best_scores) if best_scores else 1
        for round_num, score in enumerate(best_scores):
            # Create simple bar visualization
            bar_length = int((score / max_score) * 30) if max_score > 0 else 0
            bar = "█" * bar_length
            f.write(f"{round_num:<8} {score:<12.3f} {bar}\n")

    print(f"Best scores data saved to: {csv_file}")
    print(f"Best scores visualization saved to: {viz_file}")

    # Print to console
    print(f"\nBest Scores per Round{f' - {run_name}' if run_name else ''}")
    print("=" * 50)
    for round_num, score in enumerate(best_scores):
        print(f"Round {round_num}: {score:.3f}")


def plot_best_scores_matplotlib(best_scores: List[float], output_dir: Path, run_name: str = ""):
    """Plot best scores using matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping plot generation")
        return

    if not best_scores:
        print("No best scores to plot")
        return

    rounds = list(range(len(best_scores)))

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, best_scores, 'b-o', linewidth=2, markersize=8, label='Best Score')
    plt.xlabel('Debate Round', fontsize=12)
    plt.ylabel('Best Score', fontsize=12)
    plt.title(f'Best Score per Round{f" - {run_name}" if run_name else ""}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Add value annotations
    for i, score in enumerate(best_scores):
        plt.annotate(f'{score:.1f}', (i, score), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=10)

    plt.tight_layout()

    output_file = output_dir / f'best_scores{"_" + run_name if run_name else ""}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Best scores plot saved to: {output_file}")


def save_average_with_ci(all_scores_per_round: List[List[float]], output_dir: Path, run_name: str = ""):
    """Save average scores with 95% confidence intervals data and create visualization."""
    if not all_scores_per_round:
        print("No scores to save")
        return

    means = []
    lower_bounds = []
    upper_bounds = []

    for round_scores in all_scores_per_round:
        mean, lower, upper = calculate_confidence_interval(round_scores)
        means.append(mean)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    # Save as CSV
    csv_file = output_dir / f'average_scores_ci{"_" + run_name if run_name else ""}.csv'
    with open(csv_file, 'w') as f:
        f.write("Round,Mean_Score,Lower_Bound,Upper_Bound,CI_Width\n")
        for round_num, (mean, lower, upper) in enumerate(zip(means, lower_bounds, upper_bounds)):
            ci_width = upper - lower
            f.write(f"{round_num},{mean:.3f},{lower:.3f},{upper:.3f},{ci_width:.3f}\n")

    # Create simple text visualization
    viz_file = output_dir / f'average_scores_ci{"_" + run_name if run_name else ""}.txt'
    with open(viz_file, 'w') as f:
        f.write(f"Average Scores with 95% CI per Round{f' - {run_name}' if run_name else ''}\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Round':<8} {'Mean':<10} {'CI Range':<20} {'Visualization'}\n")
        f.write("-" * 80 + "\n")

        max_mean = max(means) if means else 1
        for round_num, (mean, lower, upper) in enumerate(zip(means, lower_bounds, upper_bounds)):
            # Create simple bar visualization for mean
            bar_length = int((mean / max_mean) * 20) if max_mean > 0 else 0
            bar = "█" * bar_length
            ci_range = f"[{lower:.1f}, {upper:.1f}]"
            f.write(f"{round_num:<8} {mean:<10.3f} {ci_range:<20} {bar}\n")

    print(f"Average scores with CI data saved to: {csv_file}")
    print(f"Average scores with CI visualization saved to: {viz_file}")

    # Print to console
    print(f"\nAverage Scores with 95% CI per Round{f' - {run_name}' if run_name else ''}")
    print("=" * 80)
    for round_num, (mean, lower, upper) in enumerate(zip(means, lower_bounds, upper_bounds)):
        ci_width = upper - lower
        print(f"Round {round_num}: Mean={mean:.3f}, CI=[{lower:.3f}, {upper:.3f}], Width={ci_width:.3f}")


def plot_average_with_ci_matplotlib(all_scores_per_round: List[List[float]], output_dir: Path, run_name: str = ""):
    """Plot average scores with 95% confidence intervals using matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping plot generation")
        return

    if not all_scores_per_round:
        print("No scores to plot")
        return

    rounds = list(range(len(all_scores_per_round)))
    means = []
    lower_bounds = []
    upper_bounds = []

    for round_scores in all_scores_per_round:
        mean, lower, upper = calculate_confidence_interval(round_scores)
        means.append(mean)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    plt.figure(figsize=(12, 8))

    # Plot mean line
    plt.plot(rounds, means, 'b-o', linewidth=2, markersize=8, label='Average Score')

    # Plot confidence interval
    plt.fill_between(rounds, lower_bounds, upper_bounds, alpha=0.3, color='blue',
                     label='95% Confidence Interval')

    # Add error bars
    errors = [(mean - lower, upper - mean) for mean, lower, upper in zip(means, lower_bounds, upper_bounds)]
    errors_lower, errors_upper = zip(*errors)
    plt.errorbar(rounds, means, yerr=[errors_lower, errors_upper], fmt='none',
                 color='red', alpha=0.7, capsize=5, capthick=2)

    plt.xlabel('Debate Round', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Average Score with 95% CI per Round{f" - {run_name}" if run_name else ""}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Add value annotations for means
    for i, mean in enumerate(means):
        plt.annotate(f'{mean:.1f}', (i, mean), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=10)

    plt.tight_layout()

    output_file = output_dir / f'average_scores_ci{"_" + run_name if run_name else ""}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Average scores with CI plot saved to: {output_file}")


def plot_thesis_synthesis_matplotlib(thesis_best: List[float], synthesis_best: List[float], output_dir: Path, run_name: str = ""):
    """Plot two lines: best Thesis vs best Synthesis scores per round."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping plot generation")
        return
    if not thesis_best and not synthesis_best:
        print("No scores to plot")
        return
    plt.figure(figsize=(12, 8))
    if thesis_best:
        plt.plot(range(len(thesis_best)), thesis_best, 'r-o', linewidth=2, markersize=8, label='Thesis (best)')
    if synthesis_best:
        plt.plot(range(len(synthesis_best)), synthesis_best, 'g-o', linewidth=2, markersize=8, label='Synthesis (best)')
    plt.xlabel('Debate Round', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Thesis vs Synthesis Best Score per Round{f" - {run_name}" if run_name else ""}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    # annotate
    for i, val in enumerate(thesis_best):
        if val == val:
            plt.annotate(f'{val:.1f}', (i, val), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    for i, val in enumerate(synthesis_best):
        if val == val:
            plt.annotate(f'{val:.1f}', (i, val), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    plt.tight_layout()
    output_file = output_dir / f'best_scores_thesis_synthesis{"_" + run_name if run_name else ""}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Thesis vs Synthesis best score plot saved to: {output_file}")


def plot_thesis_synthesis_ci_matplotlib(thesis_all: List[List[float]], synthesis_all: List[List[float]], output_dir: Path, run_name: str = ""):
    """Plot average + 95% CI per round for Thesis and Synthesis as two lines with shaded CI."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping plot generation")
        return
    has_thesis = any(len(r) > 0 for r in thesis_all)
    has_synth = any(len(r) > 0 for r in synthesis_all)
    if not (has_thesis or has_synth):
        print("No Thesis/Synthesis scores to plot CI")
        return

    max_len = max(len(thesis_all), len(synthesis_all))
    rounds = list(range(max_len))

    def stats(series: List[List[float]]):
        means: List[float] = []
        lowers: List[float] = []
        uppers: List[float] = []
        for r in range(max_len):
            scores = series[r] if r < len(series) else []
            mean, lower, upper = calculate_confidence_interval(scores)
            means.append(mean)
            lowers.append(lower)
            uppers.append(upper)
        return means, lowers, uppers

    thesis_means, thesis_low, thesis_up = stats(thesis_all)
    synth_means, synth_low, synth_up = stats(synthesis_all)

    plt.figure(figsize=(12, 8))
    if has_thesis:
        plt.plot(rounds, thesis_means, color='red', marker='o', linewidth=2, markersize=6, label='Thesis (mean)')
        plt.fill_between(rounds, thesis_low, thesis_up, color='red', alpha=0.2, label='Thesis 95% CI')
    if has_synth:
        plt.plot(rounds, synth_means, color='green', marker='o', linewidth=2, markersize=6, label='Synthesis (mean)')
        plt.fill_between(rounds, synth_low, synth_up, color='green', alpha=0.2, label='Synthesis 95% CI')

    plt.xlabel('Debate Round', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Thesis vs Synthesis: Average Score with 95% CI per Round{f" - {run_name}" if run_name else ""}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    output_file = output_dir / f'average_scores_ci_thesis_synthesis{"_" + run_name if run_name else ""}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Thesis vs Synthesis CI plot saved to: {output_file}")


def find_latest_debate_run(base_dir: Path) -> Optional[Path]:
    """Find the most recent debate run directory."""
    debate_dir = base_dir / "outputs" / "debate"
    if not debate_dir.exists():
        return None

    # Find all timestamp directories
    timestamp_dirs = [d for d in debate_dir.iterdir() if d.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', d.name)]
    if not timestamp_dirs:
        return None

    # Return the most recent one
    latest = max(timestamp_dirs, key=lambda x: x.stat().st_mtime)

    # Look for debate_runs subdirectory
    debate_runs_dir = latest / "debate_runs"
    if debate_runs_dir.exists():
        if any(d.is_dir() and d.name.startswith("round_") for d in debate_runs_dir.iterdir()):
            return debate_runs_dir
        # Backward-compatible support for nested run directories.
        run_dirs = [
            d for d in debate_runs_dir.iterdir()
            if d.is_dir() and any(r.is_dir() and r.name.startswith("round_") for r in d.iterdir())
        ]
        if run_dirs:
            return max(run_dirs, key=lambda x: x.stat().st_mtime)

    return latest


def infer_run_name(run_dir: Path) -> str:
    """Infer a stable run name from supported run directory layouts."""
    if run_dir.name == "debate_runs":
        return run_dir.parent.name
    return run_dir.name


def main():
    parser = argparse.ArgumentParser(description='Plot debate performance metrics')
    parser.add_argument('--run_dir', type=str, help='Path to specific debate run directory')
    parser.add_argument('--output_dir', type=str, help='Output directory for plots (defaults to run directory)')
    parser.add_argument('--project_root', type=str, default=str(PROJECT_ROOT), help='Project root directory')
    parser.add_argument('--use_matplotlib', action='store_true', help='Force matplotlib plots (matplotlib is used automatically when available)')

    args = parser.parse_args()

    project_root = Path(args.project_root)

    # Determine which run to analyze
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Error: Run directory {run_dir} does not exist")
            return
        run_name = infer_run_name(run_dir)
    else:
        # Find the latest run
        run_dir = find_latest_debate_run(project_root)
        if not run_dir:
            print("Error: No debate runs found")
            return
        run_name = infer_run_name(run_dir)

    # Use run directory as output directory unless specified otherwise
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = run_dir  # Save plots in the run directory itself

    print(f"Analyzing debate run: {run_dir}")
    print(f"Run name: {run_name}")
    print(f"Output directory: {output_dir}")

    # Extract scores
    print("\nExtracting scores from training metrics...")
    best_scores, all_scores_per_round = extract_scores_from_run(run_dir)
    thesis_best, synth_best, thesis_all, synth_all = extract_thesis_synthesis_scores_from_run(run_dir)

    if not best_scores:
        print("Error: No scores found in the run directory")
        return

    print(f"Found {len(best_scores)} rounds with scores")
    for i, scores in enumerate(all_scores_per_round):
        print(f"  Round {i}: {len(scores)} candidates, best score: {max(scores):.2f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    save_best_scores(best_scores, output_dir, run_name)
    save_average_with_ci(all_scores_per_round, output_dir, run_name)

    # Generate matplotlib plots if available (or if explicitly requested)
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating matplotlib plots...")
        plot_best_scores_matplotlib(best_scores, output_dir, run_name)
        plot_average_with_ci_matplotlib(all_scores_per_round, output_dir, run_name)
        plot_thesis_synthesis_matplotlib(thesis_best, synth_best, output_dir, run_name)
        plot_thesis_synthesis_ci_matplotlib(thesis_all, synth_all, output_dir, run_name)
    elif args.use_matplotlib and not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not available. Install with: pip install matplotlib numpy")

    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
