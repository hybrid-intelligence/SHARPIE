"""
Async WebSocket client for simulating participants in benchmarks.

Uses aiohttp for async HTTP and WebSocket connections with Django
session cookie authentication.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional
import statistics
import numpy as np

BYTES_PER_MEBIBYTE = 1024 * 1024

try:
    import aiohttp
except ImportError:
    raise ImportError(
        "aiohttp is required for participant simulation. "
        "Install with: pip install aiohttp"
    )


@dataclass
class TimingSample:
    """Single round-trip timing measurement."""
    step: int
    action_sent_time: float
    obs_received_time: float
    action_taken: int = 0
    image_size: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0

    @property
    def rtt(self) -> float:
        """Round-trip time in seconds."""
        return self.obs_received_time - self.action_sent_time


@dataclass
class ParticipantMetrics:
    """Aggregated metrics for a single participant."""
    participant_id: str
    session_cookie: str
    start_time: float = 0.0  # When participant starts (connection time)
    first_message_time: float = 0.0  # When first message is received (gameplay start)
    end_time: float = 0.0  # When episode ends
    samples: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    steps_completed: int = 0
    connected: bool = False
    bytes_sent: int = 0  # Total bytes sent to server
    bytes_received: int = 0  # Total bytes received from server

    @property
    def wait_time(self) -> float:
        """Time waiting for episode to start (all participants to connect)."""
        if self.first_message_time <= 0:
            return 0.0
        return self.first_message_time - self.start_time

    @property
    def gameplay_duration(self) -> float:
        """Actual gameplay time (excludes wait time)."""
        if self.first_message_time <= 0:
            return self.end_time - self.start_time
        return self.end_time - self.first_message_time

    @property
    def total_duration(self) -> float:
        """Total time from connection to completion."""
        return self.end_time - self.start_time

    @property
    def fps(self) -> float:
        """Steps per second during actual gameplay."""
        if self.gameplay_duration <= 0:
            return 0.0
        return self.steps_completed / self.gameplay_duration

    @property
    def rtts(self) -> list:
        """List of round-trip times."""
        return [s.rtt for s in self.samples if s.rtt > 0]

    @property
    def avg_rtt(self) -> float:
        """Average round-trip time in seconds."""
        rtts = self.rtts
        return statistics.mean(rtts) if rtts else 0.0

    @property
    def median_rtt(self) -> float:
        """Median round-trip time in seconds."""
        rtts = self.rtts
        return statistics.median(rtts) if rtts else 0.0

    @property
    def p95_rtt(self) -> float:
        """95th percentile round-trip time."""
        rtts = self.rtts
        if len(rtts) < 2:
            return self.avg_rtt
        sorted_rtts = sorted(rtts)
        idx = int(len(sorted_rtts) * 0.95)
        return sorted_rtts[min(idx, len(sorted_rtts) - 1)]

    @property
    def p99_rtt(self) -> float:
        """99th percentile round-trip time."""
        rtts = self.rtts
        if len(rtts) < 2:
            return self.avg_rtt
        sorted_rtts = sorted(rtts)
        idx = int(len(sorted_rtts) * 0.99)
        return sorted_rtts[min(idx, len(sorted_rtts) - 1)]

    @property
    def min_rtt(self) -> float:
        """Minimum round-trip time."""
        rtts = self.rtts
        return min(rtts) if rtts else 0.0

    @property
    def max_rtt(self) -> float:
        """Maximum round-trip time."""
        rtts = self.rtts
        return max(rtts) if rtts else 0.0

    @property
    def error_rate(self) -> float:
        """Proportion of steps that resulted in errors."""
        total = self.steps_completed + len(self.errors)
        if total == 0:
            return 0.0
        return len(self.errors) / total

    @property
    def upload_bandwidth_mbps(self) -> float:
        """Upload bandwidth in MiB/s during gameplay."""
        if self.gameplay_duration <= 0:
            return 0.0
        return (self.bytes_sent / self.gameplay_duration) / BYTES_PER_MEBIBYTE

    @property
    def download_bandwidth_mbps(self) -> float:
        """Download bandwidth in MiB/s during gameplay."""
        if self.gameplay_duration <= 0:
            return 0.0
        return (self.bytes_received / self.gameplay_duration) / BYTES_PER_MEBIBYTE

    def to_dict(self, include_timing_samples: bool = False) -> dict:
        """Convert metrics to dictionary for JSON serialization.
        
        Args:
            include_timing_samples: Whether to include raw timing samples (default: False)
        """
        result = {
            "participant_id": self.participant_id,
            "steps_completed": self.steps_completed,
            "total_duration_seconds": round(self.total_duration, 3),
            "wait_time_seconds": round(self.wait_time, 3),
            "gameplay_duration_seconds": round(self.gameplay_duration, 3),
            "fps": round(self.fps, 3),
            "rtt": {
                "avg_ms": round(self.avg_rtt * 1000, 2),
                "median_ms": round(self.median_rtt * 1000, 2),
                "min_ms": round(self.min_rtt * 1000, 2),
                "max_ms": round(self.max_rtt * 1000, 2),
                "p95_ms": round(self.p95_rtt * 1000, 2),
                "p99_ms": round(self.p99_rtt * 1000, 2),
            },
            "errors": self.errors,
            "error_rate": round(self.error_rate, 4),
            "connected": self.connected,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "upload_bandwidth_mbps": round(self.upload_bandwidth_mbps, 3),
            "download_bandwidth_mbps": round(self.download_bandwidth_mbps, 3),
        }
        
        if include_timing_samples:
            result["timing_samples"] = [
                {
                    "step": s.step,
                    "action_sent_time": s.action_sent_time,
                    "obs_received_time": s.obs_received_time,
                    "rtt_seconds": s.rtt,
                    "action_taken": s.action_taken,
                    "image_size": s.image_size,
                    "bytes_sent": s.bytes_sent,
                    "bytes_received": s.bytes_received,
                }
                for s in self.samples
            ]
        
        return result


class ParticipantSimulator:
    """
    Async WebSocket client simulating a participant.

    Connects to SHARPIE WebSocket endpoint with Django session authentication,
    sends random actions at configurable intervals, and collects timing metrics.
    """

    def __init__(
        self,
        participant_id: str,
        session_cookie: str,
        ws_url: str,
        role: str,
        num_steps: int = 100,
        action_interval: float = 0.01,
        connection_timeout: float = 30.0,
        verbose: bool = False,
        send_actions: bool = True,
        network_latency: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize participant simulator.

        Args:
            participant_id: Unique identifier for this participant
            session_cookie: Django session cookie value
            ws_url: WebSocket URL to connect to
            role: Agent role this participant controls
            num_steps: Number of steps to simulate
            action_interval: Seconds between sending actions
            connection_timeout: Seconds to wait for first message from runner
            verbose: Enable verbose logging
            send_actions: Whether to send actions (False for passive observers)
            network_latency: Simulated network latency in seconds (added to each message)
            seed: Random seed for reproducible action sequences
        """
        self.participant_id = participant_id
        self.session_cookie = session_cookie
        self.ws_url = ws_url
        self.role = role
        self.num_steps = num_steps
        self.action_interval = action_interval
        self.connection_timeout = connection_timeout
        self.verbose = verbose
        self.send_actions = send_actions
        self.network_latency = network_latency
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.metrics = ParticipantMetrics(
            participant_id=participant_id,
            session_cookie=session_cookie,
        )
        self._pending_action = None

    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{self.participant_id}] {message}")

    async def run(self) -> ParticipantMetrics:
        """
        Run the participant simulation.

        Returns:
            ParticipantMetrics: Collected timing metrics
        """
        self.metrics.start_time = time.time()

        try:
            await self._run_websocket()
        except Exception as e:
            self.log(f"Error: {e}")
            self.metrics.errors.append(str(e))
        finally:
            self.metrics.end_time = time.time()

        return self.metrics

    async def _run_websocket(self):
        """Run the WebSocket connection and message loop."""
        headers = {"Cookie": f"sessionid={self.session_cookie}"}
        timeout = aiohttp.ClientTimeout(total=self.connection_timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(
                self.ws_url,
                headers=headers,
            ) as ws:
                self.metrics.connected = True
                self.log(f"Connected to {self.ws_url}")

                # Wait for initial state (first broadcast from runner)
                step_count = 0
                first_message_received = False

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        received_bytes = len(msg.data.encode('utf-8'))
                        self.metrics.bytes_received += received_bytes
                        
                        data = json.loads(msg.data)

                        # Simulate network latency (server → client)
                        if self.network_latency > 0:
                            await asyncio.sleep(self.network_latency)

                        # Handle incoming broadcast messages
                        if data.get("type") == "websocket.message":
                            # Record first message time (start of gameplay)
                            if not first_message_received:
                                self.metrics.first_message_time = time.time()
                            first_message_received = True

                            # Record timing for pending action
                            if self._pending_action is not None:
                                # Extract image size for bandwidth tracking
                                image_base64 = data.get("image", "")
                                image_size_bytes = len(image_base64) if image_base64 else 0
                                
                                sample = TimingSample(
                                    step=self._pending_action["step"],
                                    action_sent_time=self._pending_action["sent_time"],
                                    obs_received_time=time.time(),
                                    action_taken=self._pending_action["action"],
                                    image_size=image_size_bytes,
                                    bytes_sent=self._pending_action.get("bytes_sent", 0),
                                    bytes_received=received_bytes,
                                )
                                self.metrics.samples.append(sample)
                                self._pending_action = None

                            step_count = data.get("step", step_count)
                            self.metrics.steps_completed = step_count

                            self.log(f"Step {step_count} received")

                            # Check if episode is complete
                            if data.get("terminated") or data.get("truncated"):
                                self.log("Episode ended")
                                break

                            # Send action if enabled (participant-controlled agents)
                            if self.send_actions:
                                await self._send_action(ws, step_count)

                        elif "error" in data:
                            self.log(f"Error: {data['error']}")
                            self.metrics.errors.append(data["error"])
                            # If error before first message, runner might not be available
                            if not first_message_received:
                                raise RuntimeError(f"Runner error: {data['error']}")

                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        self.log(f"WebSocket error: {ws.exception()}")
                        self.metrics.errors.append(str(ws.exception()))
                        break

                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        self.log("WebSocket closed")
                        if not first_message_received:
                            raise RuntimeError(
                                f"Connection closed before receiving any data. "
                                f"Ensure a runner is available with connection key."
                            )
                        break

                self.log(f"Completed {step_count} steps")

    async def _send_action(self, ws, step: int):
        """Send a random action (controlled by seed) to the server."""
        await asyncio.sleep(self.action_interval)

        # Choose random action from valid action space
        # Overcooked-style actions: 0=up, 1=down, 2=left, 3=right, 4=interact
        action_space = [0, 1, 2, 3, 4]
        random_action = int(self.rng.choice(action_space))

        # Participant sends action to be received by the runner
        # The RunConsumer forwards private messages to the runner
        action = {
            "type": "private",
            "message": "action",
            "action": random_action,
            "role": self.role,
        }

        self._pending_action = {
            "step": step,
            "sent_time": time.time(),
            "action": random_action,
        }

        message_json = json.dumps(action)
        sent_bytes = len(message_json.encode('utf-8'))
        self.metrics.bytes_sent += sent_bytes
        self._pending_action["bytes_sent"] = sent_bytes
        await ws.send_str(message_json)
        self.log(f"Sent action {random_action} for step {step}")