"""
Metrics aggregation and reporting for SHARPIE benchmarks.

Provides functions to combine metrics from multiple participants,
compute aggregate statistics, and generate reports.
"""

import csv
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
    upload_bandwidth_mbps: MetricStatistics
    download_bandwidth_mbps: MetricStatistics
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
            upload_bandwidth_mbps=compute_statistics([]),
            download_bandwidth_mbps=compute_statistics([]),
            error_rate=compute_statistics([]),
            avg_gameplay_duration_seconds=compute_statistics([]),
        )
    
    avg_fps_values = [t.metrics.avg_fps for t in trial_results]
    median_rtt_values = [t.metrics.median_rtt_ms for t in trial_results]
    upload_bandwidth_values = [t.metrics.upload_bandwidth_mbps for t in trial_results]
    download_bandwidth_values = [t.metrics.download_bandwidth_mbps for t in trial_results]
    error_rate_values = [t.metrics.error_rate for t in trial_results]
    duration_values = [t.metrics.avg_gameplay_duration_seconds for t in trial_results]
    
    return TrialStatisticsSummary(
        avg_fps=compute_statistics(avg_fps_values),
        median_rtt_ms=compute_statistics(median_rtt_values),
        upload_bandwidth_mbps=compute_statistics(upload_bandwidth_values),
        download_bandwidth_mbps=compute_statistics(download_bandwidth_values),
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

    # Bandwidth
    total_bytes_sent: int
    total_bytes_received: int
    upload_bandwidth_mbps: float
    download_bandwidth_mbps: float

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

    # Optional fields with defaults
    num_agents: int = 0
    network_latency_ms: float = 0.0
    image_size: str = "64x64"

    # Individual participant data
    participants: List[dict] = field(default_factory=list)
    
    # Webserver bytes per timestep (from Record model - runner traffic)
    # Dict mapping step_index to {from_runner_bytes, to_runner_bytes}
    webserver_bytes_by_step: dict = field(default_factory=dict)

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
            f"Bandwidth: ↑{self.upload_bandwidth_mbps:.2f} MB/s upload, ↓{self.download_bandwidth_mbps:.2f} MB/s download",
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
    include_timing_samples: bool = False,
    num_agents: int = 0,
    network_latency_ms: float = 0.0,
    image_size: str = "64x64",
    webserver_bytes_by_step: dict = None,
) -> AggregateMetrics:
    """
    Aggregate metrics from multiple participants.

    Args:
        participant_metrics: List of per-participant metrics
        benchmark_id: Unique identifier for this benchmark run
        target_steps: Target number of steps per participant
        include_timing_samples: Whether to include raw timing samples (default: False)
        num_agents: Number of AI agents (0 for participant benchmarks)
        network_latency_ms: Simulated network latency in milliseconds
        image_size: Render size as string (e.g., "64x64", "512x512")
        webserver_bytes_by_step: Dict mapping step_index to {from_runner_bytes, to_runner_bytes}

    Returns:
        AggregateMetrics: Combined statistics
    """
    if webserver_bytes_by_step is None:
        webserver_bytes_by_step = {}
    
    if not participant_metrics:
        return AggregateMetrics(
            benchmark_id=benchmark_id,
            timestamp=datetime.now().isoformat(),
            num_participants=0,
            num_agents=num_agents,
            target_steps=target_steps,
            network_latency_ms=network_latency_ms,
            image_size=image_size,
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

    # Calculate bandwidth
    total_bytes_sent = sum(m.bytes_sent for m in participant_metrics)
    total_bytes_received = sum(m.bytes_received for m in participant_metrics)
    if avg_gameplay_duration > 0:
        upload_bandwidth_mbps = (total_bytes_sent / avg_gameplay_duration) / (1024 * 1024)
        download_bandwidth_mbps = (total_bytes_received / avg_gameplay_duration) / (1024 * 1024)
    else:
        upload_bandwidth_mbps = 0.0
        download_bandwidth_mbps = 0.0

    # Error rate
    total_attempts = total_steps + total_errors
    error_rate = total_errors / total_attempts if total_attempts > 0 else 0

    return AggregateMetrics(
        benchmark_id=benchmark_id,
        timestamp=datetime.now().isoformat(),
        num_participants=len(participant_metrics),
        num_agents=num_agents,
        target_steps=target_steps,
        network_latency_ms=network_latency_ms,
        image_size=image_size,
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
        total_bytes_sent=total_bytes_sent,
        total_bytes_received=total_bytes_received,
        upload_bandwidth_mbps=round(upload_bandwidth_mbps, 3),
        download_bandwidth_mbps=round(download_bandwidth_mbps, 3),
        error_rate=error_rate,
        total_duration_seconds=round(total_duration, 3),
        avg_participant_duration_seconds=round(avg_duration, 3),
        avg_wait_time_seconds=round(avg_wait_time, 3),
        max_wait_time_seconds=round(max_wait_time, 3),
        avg_gameplay_duration_seconds=round(avg_gameplay_duration, 3),
        participants=[m.to_dict(include_timing_samples) for m in participant_metrics],
        webserver_bytes_by_step=webserver_bytes_by_step,
    )


def _percentile(values: List[float], p: float) -> float:
    """Calculate the p-th percentile of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(len(sorted_values) * (p / 100))
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


def calculate_webserver_bytes_from_records(session_id: int) -> dict:
    """
    Calculate webserver bytes sent/received per timestep from Record model data.
    
    Webserver RECEIVES:
    - From Runner: {step, observations, actions, rewards, image, terminated, truncated}
      -> State/action/reward stored in Record, image tracked separately
    
    - From Participants: Individual action messages
      -> Not stored in Record, tracked in timing_samples as bytes_sent
    
    Webserver SENDS:
    - To Runner: Batched actions from all participants
      -> Action field in Record contains all participants' actions
    
    - To Participants: Forwarded broadcast {step, image, terminated, truncated}
      -> Tracked in timing_samples as bytes_received (includes image)
    
    Args:
        session_id: Session ID to query Records from
    
    Returns:
        Dict mapping step_index to {from_runner_bytes, to_runner_bytes}
    """
    from data.models import Session, Episode, Record
    
    bytes_by_step = {}
    
    try:
        session = Session.objects.get(id=session_id)
        episodes = Episode.objects.filter(session=session)
        
        for episode in episodes:
            records = Record.objects.filter(episode=episode).order_by('step_index')
            
            for record in records:
                step = record.step_index
                
                # Bytes from runner (webserver received)
                # Record stores: observations in state, actions in action, rewards in reward
                # The runner sends the full state; webserver receives it
                from_runner_data = {
                    'observations': record.state,
                    'actions': record.action,
                    'rewards': record.reward,
                }
                from_runner_bytes = len(json.dumps(from_runner_data, separators=(',', ':')).encode('utf-8'))
                
                # Bytes to runner (webserver sent)
                # Webserver forwards batched actions from all participants
                # The action field contains all participants' actions
                to_runner_data = {
                    'actions': record.action,
                }
                to_runner_bytes = len(json.dumps(to_runner_data, separators=(',', ':')).encode('utf-8'))
                
                bytes_by_step[step] = {
                    'from_runner_bytes': from_runner_bytes,
                    'to_runner_bytes': to_runner_bytes,
                }
    except Exception as e:
        pass
    
    return bytes_by_step


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
    save_raw: bool = False,
) -> str:
    """
    Save multi-trial results to JSON with full detail.
    
    Creates:
    - Main summary file with aggregated statistics
    - (Optional) Separate raw data files with per-participant timing samples
    
    Args:
        results: Multi-trial results container
        output_dir: Directory to save results
        filename: Optional filename (default: config_name.json)
        save_raw: Whether to save raw participant data (default: False)
    
    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f"{results.config_name}.json"

    base_filename = results.config_name
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
            "upload_bandwidth_mbps": summary.upload_bandwidth_mbps.to_dict(),
            "download_bandwidth_mbps": summary.download_bandwidth_mbps.to_dict(),
            "error_rate": summary.error_rate.to_dict(),
            "avg_gameplay_duration_seconds": summary.avg_gameplay_duration_seconds.to_dict(),
        }
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    # Save raw participant data to separate files if requested
    if save_raw:
        raw_files = save_raw_participant_data(
            results.trials,
            output_dir,
            base_filename,
        )
        print(f"Raw participant data saved: {len(raw_files)} file(s)")

    return filepath


def save_raw_participant_data(
    trial_results: List[TrialResult],
    output_dir: str,
    base_filename: str,
) -> List[str]:
    """
    Save raw per-participant timing data to separate JSON files.
    
    Creates one file per trial with detailed participant data.
    
    Args:
        trial_results: List of trial results
        output_dir: Directory to save files
        base_filename: Base name for files (e.g., "benchmark_10p")
    
    Returns:
        List of file paths created
    """
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []
    
    for trial in trial_results:
        filename = f"{base_filename}_trial_{trial.trial_number}_raw.json"
        filepath = os.path.join(output_dir, filename)
        
        output = {
            "trial_number": trial.trial_number,
            "seed_used": trial.seed_used,
            "benchmark_id": trial.metrics.benchmark_id,
            "timestamp": trial.metrics.timestamp,
            "num_participants": trial.metrics.num_participants,
            "participants": trial.metrics.participants,
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        file_paths.append(filepath)
    
    return file_paths


def save_results_csv(
    metrics: AggregateMetrics,
    output_dir: str,
    filename: Optional[str] = None,
) -> str:
    """
    Save per-timestep benchmark data to CSV.
    
    Each row = one timestep from one participant/agent.
    Contains only per-step raw timing data.
    
    Columns: benchmark_id, participants, agents, seed, trial, latency_ms, image_size,
             agent_id, timestep, action_sent_time, obs_received_time, rtt_ms,
             action_taken, image_bytes, bytes_sent, bytes_received,
             webserver_sent_bytes, webserver_received_bytes
    
    Args:
        metrics: Aggregated metrics (must include timing samples from --raw-data)
        output_dir: Directory to save results
        filename: Optional filename (default: {benchmark_id}.csv)
    
    Returns:
        str: Path to saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        filename = f"{metrics.benchmark_id}.csv"
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'benchmark_id',
            'participants',
            'agents',
            'seed',
            'trial',
            'latency_ms',
            'image_size',
            'agent_id',
            'timestep',
            'action_sent_time',
            'obs_received_time',
            'rtt_ms',
            'action_taken',
            'image_bytes',
            'bytes_sent',
            'bytes_received',
            'webserver_sent_bytes',
            'webserver_received_bytes',
        ])
        
        for participant_data in metrics.participants:
            participant_id = participant_data.get('participant_id', 'unknown')
            timing_samples = participant_data.get('timing_samples', [])
            
            for sample in timing_samples:
                step = sample.get('step', 0)
                
                # Get bytes from participant (timing_samples)
                bytes_sent = sample.get('bytes_sent', 0)
                bytes_received = sample.get('bytes_received', 0)
                image_bytes = sample.get('image_size', 0)
                
                # Get bytes from/to runner (Record data)
                runner_bytes = metrics.webserver_bytes_by_step.get(step, {})
                from_runner_bytes = runner_bytes.get('from_runner_bytes', 0)
                to_runner_bytes = runner_bytes.get('to_runner_bytes', 0)
                
                # Webserver sent = to_runner (batched actions) + to_participants (includes image)
                # bytes_received from participant IS what webserver sent to participant
                webserver_sent_bytes = to_runner_bytes + bytes_received
                
                # Webserver received = from_runner (observations + image) + from_participants (actions)
                # bytes_sent from participant IS what webserver received from participant
                # image is received from runner (included in bytes_received sent to participants)
                # We need to add image size since it's not in Record's state/action/reward
                webserver_received_bytes = from_runner_bytes + image_bytes + bytes_sent
                
                writer.writerow([
                    metrics.benchmark_id,
                    metrics.num_participants,
                    metrics.num_agents,
                    '',  # seed - not available in single AggregateMetrics
                    '',  # trial - not available in single AggregateMetrics
                    metrics.network_latency_ms,
                    metrics.image_size,
                    participant_id,
                    step,
                    sample.get('action_sent_time', 0),
                    sample.get('obs_received_time', 0),
                    round(sample.get('rtt_seconds', 0) * 1000, 3),
                    sample.get('action_taken', 0),
                    image_bytes,
                    bytes_sent,
                    bytes_received,
                    webserver_sent_bytes,
                    webserver_received_bytes,
                ])
    
    return filepath


def save_multi_trial_results_csv(
    trial_results: List['TrialResult'],
    base_seed: int,
    output_dir: str,
    base_filename: str,
) -> str:
    """
    Save multi-trial per-timestep data to a single CSV file.
    
    Combines all trials into one file with trial column for filtering.
    
    Args:
        trial_results: List of TrialResult objects from orchestrator
        base_seed: Base random seed used for the benchmark
        output_dir: Directory to save results
        base_filename: Base name for the file (e.g., "benchmark_10p")
    
    Returns:
        str: Path to saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{base_filename}.csv"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'benchmark_id',
            'participants',
            'agents',
            'seed',
            'trial',
            'latency_ms',
            'image_size',
            'agent_id',
            'timestep',
            'action_sent_time',
            'obs_received_time',
            'rtt_ms',
            'action_taken',
            'image_bytes',
            'bytes_sent',
            'bytes_received',
            'webserver_sent_bytes',
            'webserver_received_bytes',
        ])
        
        for trial in trial_results:
            metrics = trial.metrics
            
            for participant_data in metrics.participants:
                participant_id = participant_data.get('participant_id', 'unknown')
                timing_samples = participant_data.get('timing_samples', [])
                
                for sample in timing_samples:
                    step = sample.get('step', 0)
                    
                    # Get bytes from participant (timing_samples)
                    bytes_sent = sample.get('bytes_sent', 0)
                    bytes_received = sample.get('bytes_received', 0)
                    image_bytes = sample.get('image_size', 0)
                    
                    # Get bytes from/to runner (Record data)
                    runner_bytes = metrics.webserver_bytes_by_step.get(step, {})
                    from_runner_bytes = runner_bytes.get('from_runner_bytes', 0)
                    to_runner_bytes = runner_bytes.get('to_runner_bytes', 0)
                    
                    # Webserver sent = to_runner (batched actions) + to_participants (includes image)
                    # bytes_received from participant IS what webserver sent to participant
                    webserver_sent_bytes = to_runner_bytes + bytes_received
                    
                    # Webserver received = from_runner (observations + image) + from_participants (actions)
                    # bytes_sent from participant IS what webserver received from participant
                    # image is received from runner (included in bytes_received sent to participants)
                    # We need to add image size since it's not in Record's state/action/reward
                    webserver_received_bytes = from_runner_bytes + image_bytes + bytes_sent
                    
                    writer.writerow([
                        metrics.benchmark_id,
                        metrics.num_participants,
                        metrics.num_agents,
                        trial.seed_used,
                        trial.trial_number,
                        metrics.network_latency_ms,
                        metrics.image_size,
                        participant_id,
                        step,
                        sample.get('action_sent_time', 0),
                        sample.get('obs_received_time', 0),
                        round(sample.get('rtt_seconds', 0) * 1000, 3),
                        sample.get('action_taken', 0),
                        image_bytes,
                        bytes_sent,
                        bytes_received,
                        webserver_sent_bytes,
                        webserver_received_bytes,
                    ])
    
    return filepath
    
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
        "Upload (MB/s)",
        "Download (MB/s)",
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
            f"{r.upload_bandwidth_mbps:.2f}",
            f"{r.download_bandwidth_mbps:.2f}",
            str(r.total_errors),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)