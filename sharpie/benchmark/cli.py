#!/usr/bin/env python
"""
Command-line interface for SHARPIE benchmarking tools.

Run from the repository root:
    sharpie-benchmark --participants 10 --steps 100 --connection-key YOUR_KEY

Usage:
    # Single benchmark with participants
    sharpie-benchmark --participants 10 --steps 100 --connection-key YOUR_KEY

    # Full participant scalability suite (1, 10, 50, 100, 250 participants)
    sharpie-benchmark --suite scalability --connection-key YOUR_KEY

    # Full AI agent scalability suite (1, 10, 50, 100, 250 AI agents)
    sharpie-benchmark --suite ai-agents --connection-key YOUR_KEY

    # Network latency suite (machine/lab/regional/global latency presets)
    sharpie-benchmark --suite network-latency --connection-key YOUR_KEY

    # Single benchmark with simulated network latency
    sharpie-benchmark -n 10 --latency global --connection-key YOUR_KEY

    # Export per-timestep data to CSV (requires --raw-data)
    sharpie-benchmark -n 10 --raw-data --format csv --connection-key YOUR_KEY

    # Custom options
    sharpie-benchmark -n 50 -s 200 --host localhost --port 8000 --connection-key YOUR_KEY -v
"""

import argparse
import asyncio
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="SHARPIE Scalability Benchmarking Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sharpie-benchmark -n 10 -s 100 -k KEY    Run 10 participants for 100 steps
  sharpie-benchmark -n 10 -s 100 -t 5 -k KEY    Run 10 participants for 100 steps, 5 trials
  sharpie-benchmark --suite scalability -k KEY  Run participant scalability suite
  sharpie-benchmark --suite ai-agents -k KEY   Run AI agent scalability suite
  sharpie-benchmark --suite network-latency -k KEY  Run network latency suite
  sharpie-benchmark --suite image-size -k KEY   Run image size suite
  sharpie-benchmark -n 10 --latency global -k KEY   Run with 200ms simulated latency
  sharpie-benchmark -n 1 --image-size 512x512 -k KEY   Run with 512x512 render size
  sharpie-benchmark -n 50 -v -k KEY        Run with verbose output
  sharpie-benchmark -n 10 -t 3 --seed 123 -k KEY   Run 3 trials with custom seed
  sharpie-benchmark -n 10 --raw-data --format csv -k KEY   Export per-timestep data to CSV

Latency presets:
  machine   - localhost (<0.1ms)
  lab       - LAN/same room (~10ms)
  national  - same country (~20ms)
  regional  - same continent (~50ms)
  global    - same planet (~200ms)

Image size presets:
  64x64     - 12 KB per frame
  128x128   - 49 KB per frame
  256x256   - 196 KB per frame
  512x512   - 786 KB per frame
  1024x1024 - 3 MB per frame

Prerequisites:
  1. Start Redis: redis-server
  2. Start webserver: cd webserver && python manage.py runserver
  3. Create a runner in Django admin with a connection key
  4. Start runner: cd runner && python manage.py --connection-key=YOUR_KEY
        """,
    )

    # Benchmark type
    parser.add_argument(
        "--suite",
        choices=["scalability", "ai-agents", "network-latency", "image-size"],
        help="Run predefined benchmark suite (scalability=participants, ai-agents=AI agents, network-latency=latency presets, image-size=render sizes)",
    )

    # Participant configuration
    parser.add_argument(
        "-n", "--participants",
        type=int,
        default=1,
        help="Number of concurrent participants (default: 1)",
    )
    parser.add_argument(
        "-s", "--steps",
        type=int,
        default=100,
        help="Number of steps per participant (default: 100)",
    )

    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="SHARPIE server hostname (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="SHARPIE server port (default: 8000)",
    )
    parser.add_argument(
        "-k", "--connection-key",
        type=str,
        default=os.environ.get("SHARPIE_RUNNER_KEY", ""),
        help="Runner connection key (or set SHARPIE_RUNNER_KEY env var)",
    )

    # Benchmark options
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Target FPS for experiment (default: 30.0)",
    )
    parser.add_argument(
        "--action-interval",
        type=float,
        default=0.01,
        help="Seconds between participant actions (default: 0.01)",
    )
    parser.add_argument(
        "--latency",
        type=str,
        choices=["machine", "lab", "regional", "global"],
        help="Simulated network latency preset: machine(<0.1ms), lab(~5ms), regional(~50ms), global(~200ms)",
    )
    parser.add_argument(
        "--image-size",
        type=str,
        choices=["64x64", "128x128", "256x256", "512x512", "1024x1024"],
        help="Image size preset for render output: 64x64(12KB), 128x128(49KB), 256x256(196KB), 512x512(786KB), 1024x1024(3MB)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/results",
        help="Output directory for results (default: benchmark/results)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep test users and data after benchmark",
    )
    parser.add_argument(
        "--runner-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for runner to pick up session (default: 30.0)",
    )
    parser.add_argument(
        "-t", "--trials",
        type=int,
        default=3,
        help="Number of trials to run per configuration (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)",
    )

    # Output options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--raw-data",
        action="store_true",
        default=False,
        help="Save raw per-participant timing data to separate files (default: False)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format: json (default) or csv (per-timestep raw data, requires --raw-data)",
    )

    args = parser.parse_args()
    
    # Validate CSV format requires raw-data
    if args.format == "csv" and not args.raw_data:
        parser.error("--format csv requires --raw-data to collect timing samples")

    # Import after argument parsing to avoid slow startup
    from sharpie.benchmark.config import (
        BenchmarkConfig, ScalabilitySuite, AIAgentConfig, AIAgentScalabilitySuite,
        NetworkLatencySuite, ImageSizeSuite, NETWORK_PRESETS, IMAGE_SIZE_PRESETS
    )
    from sharpie.benchmark.orchestrator import (
        run_benchmark, run_scalability_suite, run_ai_agent_benchmark,
        run_ai_agent_scalability_suite, run_network_latency_suite, run_image_size_suite,
        TrialResult
    )
    from sharpie.benchmark.metrics import print_comparison_table, MultiTrialResults, save_results_csv, save_multi_trial_results_csv, save_suite_merged_csv
    import json

    async def run():
        if args.suite == "scalability":
            suite = ScalabilitySuite(
                steps_per_participant=args.steps,
                host=args.host,
                port=args.port,
                runner_connection_key=args.connection_key,
                verbose=args.verbose,
                trials=args.trials,
                seed=args.seed,
                save_raw_data=args.raw_data,
                format=args.format,
            )
            results = await run_scalability_suite(suite)

            if args.format == "csv":
                # Save all trials to CSV
                for config_trials in results:
                    if config_trials:
                        config_name = config_trials[0].metrics.benchmark_id
                        output_dir = args.output_dir
                        if not os.path.isabs(output_dir):
                            output_dir = os.path.join(os.path.dirname(os.getcwd()), output_dir)
                        csv_path = save_multi_trial_results_csv(
                            config_trials,
                            suite.seed,
                            output_dir,
                            f"scalability_{config_trials[0].metrics.num_participants}p",
                        )
                        print(f"CSV saved to {csv_path}")
                # Save merged CSV for all configurations
                merged_path = save_suite_merged_csv(
                    results,
                    suite.seed,
                    output_dir,
                    "scalability",
                )
                print(f"Merged CSV saved to {merged_path}")
            elif args.json:
                # Serialize as list of config results, each containing trials
                output = []
                for config_trials in results:
                    config_name = config_trials[0].metrics.benchmark_id if config_trials else "unknown"
                    multi_trial = MultiTrialResults(
                        config_name=config_name,
                        total_trials=len(config_trials),
                        base_seed=suite.seed,
                        trials=config_trials,
                    )
                    output.append({
                        "config_name": config_name,
                        "trials": [
                            {
                                "trial_number": t.trial_number,
                                "seed_used": t.seed_used,
                                "metrics": t.metrics.to_dict(),
                            }
                            for t in config_trials
                        ],
                        "summary": multi_trial.get_summary().to_dict(),
                    })
                print(json.dumps(output, indent=2))
            else:
                print("\n" + "=" * 60)
                print("SCALABILITY BENCHMARK RESULTS")
                print("=" * 60)
                # Print summary for each config
                for config_trials in results:
                    if config_trials:
                        config_name = config_trials[0].metrics.benchmark_id
                        num_participants = config_trials[0].metrics.num_participants
                        print(f"\n{num_participants} participants ({len(config_trials)} trials):")
                        multi_trial = MultiTrialResults(
                            config_name=config_name,
                            total_trials=len(config_trials),
                            base_seed=suite.seed,
                            trials=config_trials,
                        )
                        summary = multi_trial.get_summary()
                        print(f"  Avg FPS: {summary.avg_fps.mean:.2f} ± {summary.avg_fps.stddev:.2f}")
                        print(f"  Median RTT: {summary.median_rtt_ms.mean:.2f} ± {summary.median_rtt_ms.stddev:.2f}ms")
                        print(f"  Upload BW: {summary.upload_bandwidth_mbps.mean:.2f} ± {summary.upload_bandwidth_mbps.stddev:.2f} MiB/s")
                        print(f"  Download BW: {summary.download_bandwidth_mbps.mean:.2f} ± {summary.download_bandwidth_mbps.stddev:.2f} MiB/s")

        elif args.suite == "ai-agents":
            suite = AIAgentScalabilitySuite(
                steps_per_agent=args.steps,
                host=args.host,
                port=args.port,
                runner_connection_key=args.connection_key,
                verbose=args.verbose,
                trials=args.trials,
                seed=args.seed,
                save_raw_data=args.raw_data,
                format=args.format,
            )
            results = await run_ai_agent_scalability_suite(suite)

            if args.format == "csv":
                for config_trials in results:
                    if config_trials:
                        config_name = config_trials[0].metrics.benchmark_id
                        output_dir = args.output_dir
                        if not os.path.isabs(output_dir):
                            output_dir = os.path.join(os.path.dirname(os.getcwd()), output_dir)
                        csv_path = save_multi_trial_results_csv(
                            config_trials,
                            suite.seed,
                            output_dir,
                            f"ai_agents_{config_trials[0].metrics.num_agents}a",
                        )
                        print(f"CSV saved to {csv_path}")
                # Save merged CSV for all configurations
                merged_path = save_suite_merged_csv(
                    results,
                    suite.seed,
                    output_dir,
                    "ai_agents",
                )
                print(f"Merged CSV saved to {merged_path}")
            elif args.json:
                output = []
                for config_trials in results:
                    config_name = config_trials[0].metrics.benchmark_id if config_trials else "unknown"
                    multi_trial = MultiTrialResults(
                        config_name=config_name,
                        total_trials=len(config_trials),
                        base_seed=suite.seed,
                        trials=config_trials,
                    )
                    output.append({
                        "config_name": config_name,
                        "trials": [
                            {
                                "trial_number": t.trial_number,
                                "seed_used": t.seed_used,
                                "metrics": t.metrics.to_dict(),
                            }
                            for t in config_trials
                        ],
                        "summary": multi_trial.get_summary().to_dict(),
                    })
                print(json.dumps(output, indent=2))
            else:
                print("\n" + "=" * 60)
                print("AI AGENT SCALABILITY BENCHMARK RESULTS")
                print("=" * 60)
                for config_trials in results:
                    if config_trials:
                        config_name = config_trials[0].metrics.benchmark_id
                        num_agents = config_trials[0].metrics.num_participants
                        print(f"\n{num_agents} AI agents ({len(config_trials)} trials):")
                        multi_trial = MultiTrialResults(
                            config_name=config_name,
                            total_trials=len(config_trials),
                            base_seed=suite.seed,
                            trials=config_trials,
                        )
                        summary = multi_trial.get_summary()
                        print(f"  Avg FPS: {summary.avg_fps.mean:.2f} ± {summary.avg_fps.stddev:.2f}")
                        print(f"  Median RTT: {summary.median_rtt_ms.mean:.2f} ± {summary.median_rtt_ms.stddev:.2f}ms")
                        print(f"  Upload BW: {summary.upload_bandwidth_mbps.mean:.2f} ± {summary.upload_bandwidth_mbps.stddev:.2f} MiB/s")
                        print(f"  Download BW: {summary.download_bandwidth_mbps.mean:.2f} ± {summary.download_bandwidth_mbps.stddev:.2f} MiB/s")

        elif args.suite == "network-latency":
            suite = NetworkLatencySuite(
                num_participants=args.participants,
                steps_per_participant=args.steps,
                host=args.host,
                port=args.port,
                runner_connection_key=args.connection_key,
                verbose=args.verbose,
                trials=args.trials,
                seed=args.seed,
                save_raw_data=args.raw_data,
                format=args.format,
            )
            results = await run_network_latency_suite(suite)

            if args.format == "csv":
                for config_trials in results:
                    if config_trials:
                        config_name = config_trials[0].metrics.benchmark_id
                        output_dir = args.output_dir
                        if not os.path.isabs(output_dir):
                            output_dir = os.path.join(os.path.dirname(os.getcwd()), output_dir)
                        csv_path = save_multi_trial_results_csv(
                            config_trials,
                            suite.seed,
                            output_dir,
                            f"network_latency_{config_name}",
                        )
                        print(f"CSV saved to {csv_path}")
                # Save merged CSV for all configurations
                merged_path = save_suite_merged_csv(
                    results,
                    suite.seed,
                    output_dir,
                    "network_latency",
                )
                print(f"Merged CSV saved to {merged_path}")
            elif args.json:
                output = []
                for config_trials in results:
                    config_name = config_trials[0].metrics.benchmark_id if config_trials else "unknown"
                    multi_trial = MultiTrialResults(
                        config_name=config_name,
                        total_trials=len(config_trials),
                        base_seed=suite.seed,
                        trials=config_trials,
                    )
                    output.append({
                        "config_name": config_name,
                        "trials": [
                            {
                                "trial_number": t.trial_number,
                                "seed_used": t.seed_used,
                                "metrics": t.metrics.to_dict(),
                            }
                            for t in config_trials
                        ],
                        "summary": multi_trial.get_summary().to_dict(),
                    })
                print(json.dumps(output, indent=2))
            else:
                print("\n" + "=" * 60)
                print("NETWORK LATENCY BENCHMARK RESULTS")
                print("=" * 60)
                for config_trials in results:
                    if config_trials:
                        config_name = config_trials[0].metrics.benchmark_id
                        multi_trial = MultiTrialResults(
                            config_name=config_name,
                            total_trials=len(config_trials),
                            base_seed=suite.seed,
                            trials=config_trials,
                        )
                        summary = multi_trial.get_summary()
                        print(f"\nConfig: {config_name} ({len(config_trials)} trials)")
                        print(f"  Avg FPS: {summary.avg_fps.mean:.2f} ± {summary.avg_fps.stddev:.2f}")
                        print(f"  Median RTT: {summary.median_rtt_ms.mean:.2f} ± {summary.median_rtt_ms.stddev:.2f}ms")
                        print(f"  Upload BW: {summary.upload_bandwidth_mbps.mean:.2f} ± {summary.upload_bandwidth_mbps.stddev:.2f} MiB/s")
                        print(f"  Download BW: {summary.download_bandwidth_mbps.mean:.2f} ± {summary.download_bandwidth_mbps.stddev:.2f} MiB/s")

        elif args.suite == "image-size":
            suite = ImageSizeSuite(
                num_participants=args.participants,
                steps_per_participant=args.steps,
                host=args.host,
                port=args.port,
                runner_connection_key=args.connection_key,
                verbose=args.verbose,
                trials=args.trials,
                seed=args.seed,
                save_raw_data=args.raw_data,
                format=args.format,
            )
            results = await run_image_size_suite(suite)

            if args.format == "csv":
                for config_trials in results:
                    if config_trials:
                        config_name = config_trials[0].metrics.benchmark_id
                        output_dir = args.output_dir
                        if not os.path.isabs(output_dir):
                            output_dir = os.path.join(os.path.dirname(os.getcwd()), output_dir)
                        csv_path = save_multi_trial_results_csv(
                            config_trials,
                            suite.seed,
                            output_dir,
                            f"image_size_{config_name}",
                        )
                        print(f"CSV saved to {csv_path}")
                # Save merged CSV for all configurations
                merged_path = save_suite_merged_csv(
                    results,
                    suite.seed,
                    output_dir,
                    "image_size",
                )
                print(f"Merged CSV saved to {merged_path}")
            elif args.json:
                output = []
                for config_trials in results:
                    config_name = config_trials[0].metrics.benchmark_id if config_trials else "unknown"
                    multi_trial = MultiTrialResults(
                        config_name=config_name,
                        total_trials=len(config_trials),
                        base_seed=suite.seed,
                        trials=config_trials,
                    )
                    output.append({
                        "config_name": config_name,
                        "trials": [
                            {
                                "trial_number": t.trial_number,
                                "seed_used": t.seed_used,
                                "metrics": t.metrics.to_dict(),
                            }
                            for t in config_trials
                        ],
                        "summary": multi_trial.get_summary().to_dict(),
                    })
                print(json.dumps(output, indent=2))
            else:
                print("\n" + "=" * 60)
                print("IMAGE SIZE BENCHMARK RESULTS")
                print("=" * 60)
                for config_trials in results:
                    if config_trials:
                        config_name = config_trials[0].metrics.benchmark_id
                        multi_trial = MultiTrialResults(
                            config_name=config_name,
                            total_trials=len(config_trials),
                            base_seed=suite.seed,
                            trials=config_trials,
                        )
                        summary = multi_trial.get_summary()
                        print(f"\nConfig: {config_name} ({len(config_trials)} trials)")
                        print(f"  Avg FPS: {summary.avg_fps.mean:.2f} ± {summary.avg_fps.stddev:.2f}")
                        print(f"  Median RTT: {summary.median_rtt_ms.mean:.2f} ± {summary.median_rtt_ms.stddev:.2f}ms")
                        print(f"  Upload BW: {summary.upload_bandwidth_mbps.mean:.2f} ± {summary.upload_bandwidth_mbps.stddev:.2f} MiB/s")
                        print(f"  Download BW: {summary.download_bandwidth_mbps.mean:.2f} ± {summary.download_bandwidth_mbps.stddev:.2f} MiB/s")

        else:
            # Single benchmark
            network_latency = NETWORK_PRESETS.get(args.latency, 0.0) if args.latency else 0.0
            image_size = IMAGE_SIZE_PRESETS.get(args.image_size, (64, 64, 3)) if args.image_size else (64, 64, 3)
            config = BenchmarkConfig(
                num_participants=args.participants,
                num_steps=args.steps,
                host=args.host,
                port=args.port,
                runner_connection_key=args.connection_key,
                target_fps=args.target_fps,
                action_interval=args.action_interval,
                network_latency=network_latency,
                image_size=image_size,
                verbose=args.verbose,
                output_dir=args.output_dir,
                cleanup=not args.no_cleanup,
                runner_timeout=args.runner_timeout,
                trials=args.trials,
                seed=args.seed,
                save_raw_data=args.raw_data,
                format=args.format,
            )

            trial_results = await run_benchmark(config)

            if args.format == "csv":
                output_dir = args.output_dir
                if not os.path.isabs(output_dir):
                    output_dir = os.path.join(os.path.dirname(os.getcwd()), output_dir)
                csv_path = save_multi_trial_results_csv(
                    trial_results,
                    config.seed,
                    output_dir,
                    trial_results[0].metrics.benchmark_id if trial_results else "benchmark",
                )
                print(f"CSV saved to {csv_path}")
            elif args.json:
                # Output all trials
                output = {
                    "config": {
                        "benchmark_id": trial_results[0].metrics.benchmark_id if trial_results else "unknown",
                        "total_trials": len(trial_results),
                        "base_seed": config.seed,
                    },
                    "trials": [
                        {
                            "trial_number": t.trial_number,
                            "seed_used": t.seed_used,
                            "metrics": t.metrics.to_dict(),
                        }
                        for t in trial_results
                    ],
                }
                print(json.dumps(output, indent=2))
            else:
                # Print summary for single benchmark
                if len(trial_results) == 1:
                    # Single trial, use existing format
                    print("\n" + trial_results[0].metrics.summary())
                else:
                    # Multiple trials, print summary statistics
                    print("\n" + "=" * 60)
                    print(f"BENCHMARK RESULTS ({len(trial_results)} trials)")
                    print("=" * 60)
                    multi_trial = MultiTrialResults(
                        config_name=trial_results[0].metrics.benchmark_id,
                        total_trials=len(trial_results),
                        base_seed=config.seed,
                        trials=trial_results,
                    )
                    summary = multi_trial.get_summary()
                    print(f"\nPerformance:")
                    print(f"  Avg FPS: {summary.avg_fps.mean:.2f} ± {summary.avg_fps.stddev:.2f}")
                    print(f"  Median RTT: {summary.median_rtt_ms.mean:.2f} ± {summary.median_rtt_ms.stddev:.2f}ms")
                    print(f"  Upload BW: {summary.upload_bandwidth_mbps.mean:.2f} ± {summary.upload_bandwidth_mbps.stddev:.2f} MiB/s")
                    print(f"  Download BW: {summary.download_bandwidth_mbps.mean:.2f} ± {summary.download_bandwidth_mbps.stddev:.2f} MiB/s")
                    print(f"\nAcross {len(trial_results)} trials:")
                    for i, t in enumerate(trial_results):
                        print(f"  Trial {i} (seed={t.seed_used}): FPS={t.metrics.avg_fps:.2f}, RTT={t.metrics.median_rtt_ms:.2f}ms")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()