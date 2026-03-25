"""Benchmark configuration for SHARPIE scalability testing."""

from dataclasses import dataclass, field
from typing import List, Optional
import os


# Network latency presets (in seconds)
NETWORK_PRESETS = {
    "machine": 0.0001,   # localhost/same computer (<0.1ms)
    "lab": 0.010,        # LAN/same room (<10ms)
    "national": 0.020,   # same country (<20ms)
    "regional": 0.050,   # same continent (<50ms)
    "global": 0.200,     # same planet (<200ms)
}

# Image size presets (height, width, channels)
IMAGE_SIZE_PRESETS = {
    "64x64": (64, 64, 3),           # 12 KB per frame
    "128x128": (128, 128, 3),       # 49 KB per frame
    "256x256": (256, 256, 3),       # 196 KB per frame
    "512x512": (512, 512, 3),       # 786 KB per frame
    "1024x1024": (1024, 1024, 3),   # 3 MB per frame
}


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    num_participants: int = 1
    num_steps: int = 100
    host: str = "localhost"
    port: int = 8000
    runner_connection_key: str = ""  # Connection key for the runner
    target_fps: float = 30.0
    action_interval: float = 0.01  # Seconds between actions (0 = no artificial delay)
    network_latency: float = 0.0  # Simulated network latency in seconds
    image_size: tuple = (64, 64, 3)  # Render size (height, width, channels)
    verbose: bool = False
    output_dir: str = "benchmark/results"
    cleanup: bool = True  # Delete test users after benchmark
    runner_timeout: float = 30.0  # Seconds to wait for runner to pick up session


@dataclass
class ScalabilitySuite:
    """Predefined scalability test suite."""
    participant_counts: List[int] = field(default_factory=lambda: [1, 10, 50, 100, 250])
    steps_per_participant: int = 100
    host: str = "localhost"
    port: int = 8000
    runner_connection_key: str = ""
    network_latency: float = 0.0  # Simulated network latency in seconds
    verbose: bool = False

    def get_config(self, num_participants: int) -> BenchmarkConfig:
        """Get a benchmark config for a specific participant count."""
        return BenchmarkConfig(
            num_participants=num_participants,
            num_steps=self.steps_per_participant,
            host=self.host,
            port=self.port,
            runner_connection_key=self.runner_connection_key,
            network_latency=self.network_latency,
            verbose=self.verbose,
        )


@dataclass
class NetworkLatencySuite:
    """Test suite for measuring impact of network latency."""
    latency_presets: List[str] = field(default_factory=lambda: ["machine", "lab", "national", "regional", "global"])
    num_participants: int = 10  # Fixed number of participants
    steps_per_participant: int = 100
    host: str = "localhost"
    port: int = 8000
    runner_connection_key: str = ""
    verbose: bool = False

    def get_config(self, latency_preset: str) -> BenchmarkConfig:
        """Get a benchmark config for a specific latency preset."""
        latency = NETWORK_PRESETS.get(latency_preset, 0.0)
        return BenchmarkConfig(
            num_participants=self.num_participants,
            num_steps=self.steps_per_participant,
            host=self.host,
            port=self.port,
            runner_connection_key=self.runner_connection_key,
            network_latency=latency,
            verbose=self.verbose,
        )


@dataclass
class ImageSizeSuite:
    """Test suite for measuring impact of image size on throughput."""
    image_size_presets: List[str] = field(default_factory=lambda: ["64x64", "128x128", "256x256", "512x512", "1024x1024"])
    num_participants: int = 1  # Fixed number of participants
    steps_per_participant: int = 100
    host: str = "localhost"
    port: int = 8000
    runner_connection_key: str = ""
    verbose: bool = False

    def get_config(self, image_size_preset: str) -> BenchmarkConfig:
        """Get a benchmark config for a specific image size preset."""
        image_size = IMAGE_SIZE_PRESETS.get(image_size_preset, (64, 64, 3))
        return BenchmarkConfig(
            num_participants=self.num_participants,
            num_steps=self.steps_per_participant,
            host=self.host,
            port=self.port,
            runner_connection_key=self.runner_connection_key,
            image_size=image_size,
            verbose=self.verbose,
        )


# Benchmark experiment configuration
BENCHMARK_EXPERIMENT = {
    "name": "Scalability Benchmark",
    "link": "benchmark-scalability",
    "short_description": "Infrastructure overhead benchmark",
    "long_description": "Measures SHARPIE's capacity for hosting large-scale crowdsourced studies.",
    "environment_file": "noop_environment.py",
    "target_fps": 30.0,
    "wait_for_inputs": False,  # Wait for participant actions
    "number_of_episodes": 1,
}

# AI agent benchmark experiment configuration
AI_AGENT_EXPERIMENT = {
    "name": "AI Agent Scalability Benchmark",
    "link": "benchmark-ai-agents",
    "short_description": "AI agent overhead benchmark",
    "long_description": "Measures SHARPIE's capacity for running many AI agents with policies.",
    "environment_file": "noop_environment.py",
    "target_fps": 30.0,
    "wait_for_inputs": False,
    "number_of_episodes": 1,
}

@dataclass
class AIAgentConfig:
    """Configuration for AI agent benchmark run."""
    num_agents: int = 1
    num_steps: int = 100
    host: str = "localhost"
    port: int = 8000
    runner_connection_key: str = ""
    target_fps: float = 30.0
    verbose: bool = False
    output_dir: str = "benchmark/results"
    cleanup: bool = True
    runner_timeout: float = 30.0


@dataclass
class AIAgentScalabilitySuite:
    """Predefined AI agent scalability test suite."""
    agent_counts: List[int] = field(default_factory=lambda: [1, 10, 50, 100, 250])
    steps_per_agent: int = 100
    host: str = "localhost"
    port: int = 8000
    runner_connection_key: str = ""
    verbose: bool = False

    def get_config(self, num_agents: int) -> AIAgentConfig:
        """Get a benchmark config for a specific agent count."""
        return AIAgentConfig(
            num_agents=num_agents,
            num_steps=self.steps_per_agent,
            host=self.host,
            port=self.port,
            runner_connection_key=self.runner_connection_key,
            verbose=self.verbose,
        )


