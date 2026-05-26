"""
Metrics aggregation and reporting for SHARPIE benchmarks.

Provides functions to combine metrics from multiple participants,
compute aggregate statistics, and generate reports.
"""

import json
import os
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional
import numpy as np

from .participant_simulator import ParticipantMetrics


@dataclass
class TrialResult:
    """Container for a single trial result."""
    trial_number: int
    metrics: 'AggregateMetrics'
    seed_used: int


@dataclass
class MetricStatistics:
    """Statistics for a single metric across trials."""
    mean: float
    median: float
    stddev: float
    min_val: float
    max_val: float
    
    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 3),
            "median": round(self.median, 3),
            "stddev": round(self.stddev, 3),
            "min": round(self.min_val, 3),
            "max": round(self.max_val, 3),
        }


@dataclass
class MultiTrialResults:
    """Container for multi-trial benchmark results."""
    config_name: str
    total_trials: int
    base_seed: int
    trials: List[TrialResult]
    
    def get_summary(self) -> 'TrialStatisticsSummary':
        """Compute statistics across all trials."""
        return compute_multi_trial_statistics(self.trials)


@dataclass
class TrialStatisticsSummary:
    """Summary statistics across multiple trials."""
    avg_fps: MetricStatistics
    median_rtt_ms: MetricStatistics
    total_messages_per_second: MetricStatistics
    error_rate: MetricStatistics
    avg_gameplay_duration_seconds: MetricStatistics


def compute_statistics(values: List[float]) -> MetricStatistics:
    """Compute statistics across values."""
    if not values:
        return MetricStatistics(mean=0, median=0, stddev=0, min_val=0, max_val=0)
    
    return MetricStatistics(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        stddev=float(np.std(values)),
        min_val=float(np.min(values)),
        max_val=float(np.max(values))
    )


def compute_multi_trial_statistics(trial_results: List[TrialResult]) -> TrialStatisticsSummary:
    """Compute statistics across multiple trials."""
    if not trial_results:
        return TrialStatisticsSummary(
            avg_fps=compute_statistics([]),
            median_rtt_ms=compute_statistics([]),
            total_messages_per_second=compute_statistics([]),
            error_rate=compute_statistics([]),
            avg_gameplay_duration_seconds=compute_statistics([]),
        )
    
    avg_fps_values = [t.metrics.avg_fps for t in trial_results]
    median_rtt_values = [t.metrics.median_rtt_ms for t in trial_results]
    throughput_values = [t.metrics.total_messages_per_second for t in trial_results]
    error_rate_values = [t.metrics.error_rate for t in trial_results]
    duration_values = [t.metrics.avg_gameplay_duration_seconds for t in trial_results]
    
    return TrialStatisticsSummary(
        avg_fps=compute_statistics(avg_fps_values),
        median_rtt_ms=compute_statistics(median_rtt_values),
        total_messages_per_second=compute_statistics(throughput_values),
        error_rate=compute_statistics(error_rate_values),
        avg_gameplay_duration_seconds=compute_statistics(duration_values),
    )


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all participants."""
    benchmark_id: str
    timestamp: str
    num_participants: int
    target_steps: int
    total_steps_completed: int
    total_errors: int

    # FPS statistics (based on gameplay duration, excluding wait time)
    avg_fps: float
    min_fps: float
    max_fps: float
    median_fps: float
    std_fps: float

    # RTT statistics (in milliseconds)
    avg_rtt_ms: float
    median_rtt_ms: float
    min_rtt_ms: float
    max_rtt_ms: float
    p95_rtt_ms: float
    p99_rtt_ms: float
    std_rtt_ms: float

    # Throughput
    total_messages_per_second: float

    # Error rates
    error_rate: float

    # Duration (total time including wait)
    total_duration_seconds: float
    avg_participant_duration_seconds: float

    # Wait time (time waiting for all participants to connect)
    avg_wait_time_seconds: float
    max_wait_time_seconds: float

    # Gameplay duration (actual time spent playing)
    avg_gameplay_duration_seconds: float

    # Individual participant data
    participants: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"=== Benchmark Results: {self.benchmark_id} ===",
            f"Timestamp: {self.timestamp}",
            f"Participants: {self.num_participants}",
            f"Target steps: {self.target_steps}",
            f"Steps completed: {self.total_steps_completed}",
            "",
            "=== Performance ===",
            f"FPS: avg={self.avg_fps:.2f}, min={self.min_fps:.2f}, max={self.max_fps:.2f}, median={self.median_fps:.2f}",
            f"RTT (ms): avg={self.avg_rtt_ms:.2f}, median={self.median_rtt_ms:.2f}, p95={self.p95_rtt_ms:.2f}, p99={self.p99_rtt_ms:.2f}",
            f"Throughput: {self.total_messages_per_second:.2f} messages/sec",
            "",
            "=== Timing ===",
            f"Avg wait time: {self.avg_wait_time_seconds:.3f}s (waiting for all participants)",
            f"Avg gameplay duration: {self.avg_gameplay_duration_seconds:.3f}s",
            f"Total duration: {self.total_duration_seconds:.2f}s",
            "",
            "=== Reliability ===",
            f"Total errors: {self.total_errors}",
            f"Error rate: {self.error_rate:.4f}",
        ]
        return "\n".join(lines)


def aggregate_metrics(
    participant_metrics: List[ParticipantMetrics],
    benchmark_id: str,
    target_steps: int,
) -> AggregateMetrics:
    """
    Aggregate metrics from multiple participants.

    Args:
        participant_metrics: List of per-participant metrics
        benchmark_id: Unique identifier for this benchmark run
        target_steps: Target number of steps per participant

    Returns:
        AggregateMetrics: Combined statistics
    """
    if not participant_metrics:
        return AggregateMetrics(
            benchmark_id=benchmark_id,
            timestamp=datetime.now().isoformat(),
            num_participants=0,
            target_steps=target_steps,
            total_steps_completed=0,
            total_errors=0,
            avg_fps=0.0,
            min_fps=0.0,
            max_fps=0.0,
            median_fps=0.0,
            std_fps=0.0,
            avg_rtt_ms=0.0,
            median_rtt_ms=0.0,
            min_rtt_ms=0.0,
            max_rtt_ms=0.0,
            p95_rtt_ms=0.0,
            p99_rtt_ms=0.0,
            std_rtt_ms=0.0,
            total_messages_per_second=0.0,
            error_rate=0.0,
            total_duration_seconds=0.0,
            avg_participant_duration_seconds=0.0,
            avg_wait_time_seconds=0.0,
            max_wait_time_seconds=0.0,
            avg_gameplay_duration_seconds=0.0,
        )

    # Collect FPS values
    fps_values = [m.fps for m in participant_metrics]

    # Collect all RTTs across all participants
    all_rtts = []
    for m in participant_metrics:
        all_rtts.extend(m.rtts)

    # Convert RTTs to milliseconds
    rtt_ms_values = [r * 1000 for r in all_rtts] if all_rtts else [0]

    # Calculate totals
    total_steps = sum(m.steps_completed for m in participant_metrics)
    total_errors = sum(len(m.errors) for m in participant_metrics)
    total_duration = max(m.total_duration for m in participant_metrics)
    avg_duration = statistics.mean([m.total_duration for m in participant_metrics])

    # Calculate wait times and gameplay durations
    wait_times = [m.wait_time for m in participant_metrics]
    gameplay_durations = [m.gameplay_duration for m in participant_metrics]
    avg_wait_time = statistics.mean(wait_times) if wait_times else 0.0
    max_wait_time = max(wait_times) if wait_times else 0.0
    avg_gameplay_duration = statistics.mean(gameplay_durations) if gameplay_durations else 0.0

    # Calculate throughput (messages per second system-wide)
    # Each step involves: participant sends action, runner broadcasts state
    # So ~2 messages per step per participant
    total_messages = total_steps * 2
    throughput = total_messages / total_duration if total_duration > 0 else 0

    # Error rate
    total_attempts = total_steps + total_errors
    error_rate = total_errors / total_attempts if total_attempts > 0 else 0

    return AggregateMetrics(
        benchmark_id=benchmark_id,
        timestamp=datetime.now().isoformat(),
        num_participants=len(participant_metrics),
        target_steps=target_steps,
        total_steps_completed=total_steps,
        total_errors=total_errors,
        avg_fps=statistics.mean(fps_values) if fps_values else 0,
        min_fps=min(fps_values) if fps_values else 0,
        max_fps=max(fps_values) if fps_values else 0,
        median_fps=statistics.median(fps_values) if fps_values else 0,
        std_fps=statistics.stdev(fps_values) if len(fps_values) > 1 else 0,
        avg_rtt_ms=statistics.mean(rtt_ms_values) if rtt_ms_values else 0,
        median_rtt_ms=statistics.median(rtt_ms_values) if rtt_ms_values else 0,
        min_rtt_ms=min(rtt_ms_values) if rtt_ms_values else 0,
        max_rtt_ms=max(rtt_ms_values) if rtt_ms_values else 0,
        p95_rtt_ms=_percentile(rtt_ms_values, 95),
        p99_rtt_ms=_percentile(rtt_ms_values, 99),
        std_rtt_ms=statistics.stdev(rtt_ms_values) if len(rtt_ms_values) > 1 else 0,
        total_messages_per_second=throughput,
        error_rate=error_rate,
        total_duration_seconds=round(total_duration, 3),
        avg_participant_duration_seconds=round(avg_duration, 3),
        avg_wait_time_seconds=round(avg_wait_time, 3),
        max_wait_time_seconds=round(max_wait_time, 3),
        avg_gameplay_duration_seconds=round(avg_gameplay_duration, 3),
        participants=[m.to_dict() for m in participant_metrics],
    )


def _percentile(values: List[float], p: float) -> float:
    """Calculate the p-th percentile of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(len(sorted_values) * (p / 100))
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


def save_results(
    metrics: AggregateMetrics,
    output_dir: str,
    filename: Optional[str] = None,
) -> str:
    """
    Save benchmark results to a JSON file.

    Args:
        metrics: Aggregated metrics to save
        output_dir: Directory to save results
        filename: Optional filename (default: benchmark_id_timestamp.json)

    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f"{metrics.benchmark_id}.json"

    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    return filepath


def save_multi_trial_results(
    results: MultiTrialResults,
    output_dir: str,
    filename: Optional[str] = None,
) -> str:
    """
    Save multi-trial results to JSON with full detail.
    
    Output includes:
    - Per-trial aggregated metrics (FPS, RTT, throughput, etc.)
    - Per-trial detailed participant data (timing samples, errors, etc.)
    - Cross-trial summary statistics

    Args:
        results: Multi-trial results container
        output_dir: Directory to save results
        filename: Optional filename (default: config_name.json)

    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f"{results.config_name}.json"

    filepath = os.path.join(output_dir, filename)

    # Compute summary statistics
    summary = results.get_summary()

    output = {
        "config": {
            "config_name": results.config_name,
            "total_trials": results.total_trials,
            "base_seed": results.base_seed,
        },
        "trials": [
            {
                "trial_number": t.trial_number,
                "seed_used": t.seed_used,
                "metrics": t.metrics.to_dict(),
            }
            for t in results.trials
        ],
        "summary": {
            "avg_fps": summary.avg_fps.to_dict(),
            "median_rtt_ms": summary.median_rtt_ms.to_dict(),
            "total_messages_per_second": summary.total_messages_per_second.to_dict(),
            "error_rate": summary.error_rate.to_dict(),
            "avg_gameplay_duration_seconds": summary.avg_gameplay_duration_seconds.to_dict(),
        }
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    return filepath


def print_comparison_table(results: List[AggregateMetrics]) -> str:
    """
    Generate a comparison table for multiple benchmark runs.

    Args:
        results: List of aggregate metrics from different participant counts

    Returns:
        str: Formatted comparison table
    """
    headers = [
        "Participants",
        "Avg FPS",
        "Median RTT (ms)",
        "P95 RTT (ms)",
        "Throughput (msg/s)",
        "Errors",
    ]

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")

    for r in results:
        row = [
            str(r.num_participants),
            f"{r.avg_fps:.2f}",
            f"{r.median_rtt_ms:.2f}",
            f"{r.p95_rtt_ms:.2f}",
            f"{r.total_messages_per_second:.2f}",
            str(r.total_errors),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)