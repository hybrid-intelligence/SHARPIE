#!/usr/bin/env python
"""
Command-line interface for SHARPIE benchmarking tools.

Run from the webserver directory:
    cd webserver
    python ../benchmark/cli.py --participants 10 --steps 100 --connection-key YOUR_KEY

Usage:
    # Single benchmark with participants
    python ../benchmark/cli.py --participants 10 --steps 100 --connection-key YOUR_KEY

    # Full participant scalability suite (1, 10, 50, 100, 250 participants)
    python ../benchmark/cli.py --suite scalability --connection-key YOUR_KEY

    # Full AI agent scalability suite (1, 10, 50, 100, 250 AI agents)
    python ../benchmark/cli.py --suite ai-agents --connection-key YOUR_KEY

    # Network latency suite (machine/lab/regional/global latency presets)
    python ../benchmark/cli.py --suite network-latency --connection-key YOUR_KEY

    # Single benchmark with simulated network latency
    python ../benchmark/cli.py -n 10 --latency global --connection-key YOUR_KEY

    # Custom options
    python ../benchmark/cli.py -n 50 -s 200 --host localhost --port 8000 --connection-key YOUR_KEY -v
"""

import argparse
import asyncio
import sys
import os

# Verify we're in the webserver directory
_cwd = os.path.basename(os.getcwd())
if _cwd != 'webserver':
    print("ERROR: Benchmark must be run from the 'webserver' directory.")
    print(f"Current directory: {os.getcwd()}")
    print("\nRun:")
    print("  cd webserver")
    print("  python ../benchmark/cli.py --help")
    sys.exit(1)

# Add project root to sys.path for benchmark module imports
# (we're in webserver, so parent is project root)
_project_root = os.path.dirname(os.getcwd())
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    parser = argparse.ArgumentParser(
        description="SHARPIE Scalability Benchmarking Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmark.cli -n 10 -s 100 -k KEY    Run 10 participants for 100 steps
  python -m benchmark.cli --suite scalability -k KEY  Run participant scalability suite
  python -m benchmark.cli --suite ai-agents -k KEY   Run AI agent scalability suite
  python -m benchmark.cli --suite network-latency -k KEY  Run network latency suite
  python -m benchmark.cli --suite image-size -k KEY   Run image size suite
  python -m benchmark.cli -n 10 --latency global -k KEY   Run with 200ms simulated latency
  python -m benchmark.cli -n 1 --image-size 512x512 -k KEY   Run with 512x512 render size
  python -m benchmark.cli -n 50 -v -k KEY        Run with verbose output

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
        default=24.0,
        help="Target FPS for experiment (default: 24.0)",
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

    args = parser.parse_args()

    # Import after argument parsing to avoid slow startup
    from benchmark.config import (
        BenchmarkConfig, ScalabilitySuite, AIAgentConfig, AIAgentScalabilitySuite,
        NetworkLatencySuite, ImageSizeSuite, NETWORK_PRESETS, IMAGE_SIZE_PRESETS
    )
    from benchmark.orchestrator import (
        run_benchmark, run_scalability_suite, run_ai_agent_benchmark,
        run_ai_agent_scalability_suite, run_network_latency_suite, run_image_size_suite
    )
    from benchmark.metrics import print_comparison_table

    async def run():
        if args.suite == "scalability":
            suite = ScalabilitySuite(
                steps_per_participant=args.steps,
                host=args.host,
                port=args.port,
                runner_connection_key=args.connection_key,
                verbose=args.verbose,
            )
            results = await run_scalability_suite(suite)

            if args.json:
                import json
                print(json.dumps([r.to_dict() for r in results], indent=2))
            else:
                print("\n" + "=" * 60)
                print("SCALABILITY BENCHMARK RESULTS")
                print("=" * 60)
                print(print_comparison_table(results))

        elif args.suite == "ai-agents":
            suite = AIAgentScalabilitySuite(
                steps_per_agent=args.steps,
                host=args.host,
                port=args.port,
                runner_connection_key=args.connection_key,
                verbose=args.verbose,
            )
            results = await run_ai_agent_scalability_suite(suite)

            if args.json:
                import json
                print(json.dumps([r.to_dict() for r in results], indent=2))
            else:
                print("\n" + "=" * 60)
                print("AI AGENT SCALABILITY BENCHMARK RESULTS")
                print("=" * 60)
                print(print_comparison_table(results))

        elif args.suite == "network-latency":
            suite = NetworkLatencySuite(
                num_participants=args.participants,
                steps_per_participant=args.steps,
                host=args.host,
                port=args.port,
                runner_connection_key=args.connection_key,
                verbose=args.verbose,
            )
            results = await run_network_latency_suite(suite)

            if args.json:
                import json
                print(json.dumps([r.to_dict() for r in results], indent=2))
            else:
                print("\n" + "=" * 60)
                print("NETWORK LATENCY BENCHMARK RESULTS")
                print("=" * 60)
                print(print_comparison_table(results))

        elif args.suite == "image-size":
            suite = ImageSizeSuite(
                num_participants=args.participants,
                steps_per_participant=args.steps,
                host=args.host,
                port=args.port,
                runner_connection_key=args.connection_key,
                verbose=args.verbose,
            )
            results = await run_image_size_suite(suite)

            if args.json:
                import json
                print(json.dumps([r.to_dict() for r in results], indent=2))
            else:
                print("\n" + "=" * 60)
                print("IMAGE SIZE BENCHMARK RESULTS")
                print("=" * 60)
                print(print_comparison_table(results))

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
            )

            metrics = await run_benchmark(config)

            if args.json:
                import json
                print(json.dumps(metrics.to_dict(), indent=2))
            else:
                print("\n" + metrics.summary())

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()