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

from .participant_simulator import ParticipantMetrics


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