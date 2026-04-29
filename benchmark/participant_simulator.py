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
    image_size: int = 0

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

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for JSON serialization."""
        return {
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
            "errors": len(self.errors),
            "error_rate": round(self.error_rate, 4),
            "connected": self.connected,
        }


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
                                sample = TimingSample(
                                    step=self._pending_action["step"],
                                    action_sent_time=self._pending_action["sent_time"],
                                    obs_received_time=time.time(),
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
        """Send a hardcoded no-op action (action=0) to the server."""
        await asyncio.sleep(self.action_interval)

        # Participant sends action to be received by the runner
        # The RunConsumer forwards private messages to the runner
        action = {
            "type": "private",
            "message": "action",
            "action": 0,  # Default "no-op" action
            "role": self.role,
        }

        self._pending_action = {
            "step": step,
            "sent_time": time.time(),
        }

        await ws.send_json(action)
        self.log(f"Sent action for step {step}")