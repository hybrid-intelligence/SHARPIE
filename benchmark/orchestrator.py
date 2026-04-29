"""
Orchestrator for SHARPIE benchmark execution.

Manages the complete benchmark lifecycle:
1. Setup: Create test users, participants, experiments, sessions
2. Execution: Run concurrent participant simulators
3. Cleanup: Remove test data

Prerequisites:
- SHARPIE webserver running (run `python manage.py runserver` from webserver/)
- Redis server running (redis-server)
- A runner registered in Django admin with a connection key
- The runner actively polling (run `python manage.py --connection-key=KEY` from runner/)

The benchmark flow:
1. Orchestrator creates session with status='ready'
2. Participants connect via WebSocket
3. When all participants connect, status becomes 'pending'
4. Runner picks up the pending session and starts environment
5. Environment sends observations, participants send actions
6. Metrics are collected and reported
"""

import asyncio
import os
import shutil
import sys
import uuid
from typing import List, Tuple, Optional

# Path setup to avoid conflicts between:
# - top-level 'runner' directory and 'webserver/runner' Django app
# Must be done before Django setup
_webserver_path = os.path.abspath(os.getcwd())  # Should be webserver directory
_project_root = os.path.dirname(_webserver_path)
_benchmark_path = os.path.join(_project_root, 'benchmark')

# Clean sys.path: remove project root to avoid Django app conflicts
_cleaned_path = []
for p in sys.path:
    norm_p = os.path.normpath(os.path.abspath(p))
    if norm_p != os.path.normpath(_project_root):
        _cleaned_path.append(p)
sys.path[:] = _cleaned_path

# Add webserver for Django (mysite.settings) and benchmark for imports
if _webserver_path not in sys.path:
    sys.path.insert(0, _webserver_path)
if _benchmark_path not in sys.path:
    sys.path.insert(0, _benchmark_path)

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
import django
django.setup()

from django.contrib.auth.models import User
from django.test import Client
from asgiref.sync import sync_to_async

from accounts.models import Participant, Consent
from experiment.models import Experiment, Environment, Agent, Policy
from data.models import Session
from runner.models import Runner

from .config import BenchmarkConfig, BENCHMARK_EXPERIMENT, AIAgentConfig, AI_AGENT_EXPERIMENT, NetworkLatencySuite, ImageSizeSuite
from .participant_simulator import ParticipantSimulator, ParticipantMetrics
from .metrics import aggregate_metrics, save_results, AggregateMetrics

# Absolute path to benchmark results directory
_BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_BENCHMARK_DIR, 'results')


class BenchmarkOrchestrator:
    """
    Orchestrates benchmark setup, execution, and cleanup.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.test_users: List[User] = []
        self.test_participants: List[Participant] = []
        self.test_session: Optional[Session] = None
        self.experiment: Optional[Experiment] = None
        self.room_id: str = ""
        # Build benchmark ID with latency and image size info
        latency_suffix = ""
        if config.network_latency > 0:
            latency_ms = int(config.network_latency * 1000)
            latency_suffix = f"_l{latency_ms}ms"
        # Add image size suffix if not default (64x64)
        image_suffix = ""
        default_size = (64, 64, 3)
        if config.image_size != default_size:
            h, w, _ = config.image_size
            image_suffix = f"_{h}x{w}"
        self.benchmark_id: str = f"benchmark_{config.num_participants}p{latency_suffix}{image_suffix}_{uuid.uuid4().hex[:8]}"
        self.runner: Optional[Runner] = None

    async def setup(self) -> None:
        """
        Set up benchmark infrastructure.

        Creates:
        - Test users and participants
        - Benchmark experiment (if not exists)
        - Session for the test
        """
        print(f"[{self.benchmark_id}] Setting up benchmark...")

        # Verify runner is available
        if self.config.runner_connection_key:
            await self._verify_runner()
        else:
            print(f"[{self.benchmark_id}] Warning: No runner connection key provided. "
                  "Make sure a runner is available to pick up the session.")

        # Ensure no-op environment exists
        await self._setup_environment()

        # Ensure benchmark experiment exists
        await self._setup_experiment()

        # Create test users and participants
        await self._create_test_participants()

        # Create session
        await self._create_session()

        print(f"[{self.benchmark_id}] Setup complete")

    @sync_to_async
    def _verify_runner(self):
        """Verify that a runner with the given connection key exists and is idle."""
        try:
            runner = Runner.objects.get(connection_key=self.config.runner_connection_key)
            if runner.status != 'idle':
                raise RuntimeError(
                    f"Runner '{runner.connection_key}' is not idle (status: {runner.status}). "
                    "Wait for the runner to become available."
                )
            self.runner = runner
            print(f"[{self.benchmark_id}] Verified runner: {runner.connection_key} (status: {runner.status})")
        except Runner.DoesNotExist:
            raise RuntimeError(
                f"No runner found with connection key '{self.config.runner_connection_key}'. "
                "Create a runner in Django admin first."
            )

    @sync_to_async
    def _setup_environment(self):
        """Create no-op environment in database with metadata for max_steps and render_size."""
        runner_env_path = os.path.join(
            os.path.dirname(__file__), '..', 'runner', 'noop_environment.py'
        )
        benchmark_env_path = os.path.join(
            os.path.dirname(__file__), 'noop_environment.py'
        )

        # Copy environment file to runner directory if not exists
        if not os.path.exists(runner_env_path):
            os.makedirs(os.path.dirname(runner_env_path), exist_ok=True)
            shutil.copy(benchmark_env_path, runner_env_path)

        # Create or get environment with metadata for max_steps and render_size
        env, created = Environment.objects.get_or_create(
            name="NoOp Benchmark Environment",
            defaults={
                "description": "Minimal environment for measuring infrastructure overhead",
                "filepaths": {"environment": "noop_environment.py"},
                "metadata": {
                    "max_steps": self.config.num_steps,
                    "render_size": list(self.config.image_size),
                },
            }
        )
        # Update metadata if environment already exists
        if not created:
            env.metadata = {
                "max_steps": self.config.num_steps,
                "render_size": list(self.config.image_size),
            }
            env.save()
        self.environment = env

    @sync_to_async
    def _setup_experiment(self):
        """Create or get benchmark experiment with agents for each participant."""
        # Create agents - one for each participant slot
        self.agents = []
        for i in range(self.config.num_participants):
            role = f"participant_{i}"
            agent, created = Agent.objects.get_or_create(
                role=role,
                defaults={
                    "name": f"Benchmark Participant Agent {i}",
                    "participant": True,
                    "inputs_type": "actions",
                    "keyboard_inputs": {"default": 0},
                }
            )
            self.agents.append(agent)

        # Create or get experiment
        experiment, created = Experiment.objects.get_or_create(
            link=BENCHMARK_EXPERIMENT["link"],
            defaults={
                "name": BENCHMARK_EXPERIMENT["name"],
                "short_description": BENCHMARK_EXPERIMENT["short_description"],
                "long_description": BENCHMARK_EXPERIMENT["long_description"],
                "environment": self.environment,
                "target_fps": BENCHMARK_EXPERIMENT["target_fps"],
                "wait_for_inputs": BENCHMARK_EXPERIMENT["wait_for_inputs"],
                "number_of_episodes": BENCHMARK_EXPERIMENT["number_of_episodes"],
            }
        )

        # Clear any existing agents and add the new ones for this benchmark
        experiment.agents.clear()
        for agent in self.agents:
            experiment.agents.add(agent)

        self.experiment = experiment
        # Store agent roles for participant assignment
        self.agent_roles = [agent.role for agent in self.agents]

    @sync_to_async
    def _create_test_participants(self):
        """Create test users and participants."""
        # Get or create consent
        consent, _ = Consent.objects.get_or_create(
            name="Benchmark Consent",
            defaults={
                "explanation_text": "Automated benchmark testing consent.",
            }
        )

        for i in range(self.config.num_participants):
            username = f"benchmark_user_{self.benchmark_id}_{i}"

            user = User.objects.create_user(
                username=username,
                password="benchmark_password",
            )

            participant = Participant.objects.create(
                user=user,
                external_id=f"bench_{self.benchmark_id}_{i}",
                consent=consent,
            )

            self.test_users.append(user)
            self.test_participants.append(participant)

    @sync_to_async
    def _create_session(self):
        """Create session for benchmark."""
        self.room_id = f"room_{self.benchmark_id}"

        session = Session.objects.create(
            experiment=self.experiment,
            room=self.room_id,
            status="ready",
        )

        # Add all participants to the session
        for participant in self.test_participants:
            session.participants.add(participant)

        session.save()

        self.test_session = session

    @sync_to_async
    def _get_session_cookies(self) -> List[Tuple[str, str, str]]:
        """
        Get authenticated session cookies for all test participants.

        Returns:
            List of (participant_id, session_cookie, role) tuples
        """
        from django.contrib.sessions.backends.db import SessionStore

        cookies = []

        for i, (user, participant) in enumerate(zip(self.test_users, self.test_participants)):
            # Create a new session for this user
            session = SessionStore()
            session['_auth_user_id'] = str(user.id)
            session['_auth_user_backend'] = 'django.contrib.auth.backends.ModelBackend'
            session['_auth_user_hash'] = user.get_session_auth_hash()
            # Assign unique role to each participant
            role = self.agent_roles[i]
            session['role'] = role
            session['session'] = self.test_session.id
            session.save()

            session_key = session.session_key
            cookies.append((str(participant.id), session_key, role))

        return cookies

    async def run(self) -> AggregateMetrics:
        """
        Execute the benchmark.

        Returns:
            AggregateMetrics: Combined metrics from all participants
        """
        print(f"[{self.benchmark_id}] Starting benchmark with {self.config.num_participants} participants...")

        # Get session cookies with roles
        cookies = await self._get_session_cookies()

        # Build WebSocket URL (all participants connect to the same room)
        ws_url = (
            f"ws://{self.config.host}:{self.config.port}"
            f"/experiment/{self.experiment.link}/run/{self.room_id}"
        )

        # Create participant simulators
        simulators = []
        for participant_id, session_cookie, role in cookies:
            simulator = ParticipantSimulator(
                participant_id=participant_id,
                session_cookie=session_cookie,
                ws_url=ws_url,
                role=role,
                num_steps=self.config.num_steps,
                action_interval=self.config.action_interval,
                connection_timeout=self.config.runner_timeout,
                verbose=self.config.verbose,
                network_latency=self.config.network_latency,
            )
            simulators.append(simulator)

        # Run all simulators concurrently
        tasks = [sim.run() for sim in simulators]
        participant_metrics: List[ParticipantMetrics] = await asyncio.gather(*tasks)

        # Aggregate metrics
        metrics = aggregate_metrics(
            participant_metrics,
            benchmark_id=self.benchmark_id,
            target_steps=self.config.num_steps,
        )

        print(f"[{self.benchmark_id}] Benchmark complete")
        print(metrics.summary())

        # Save results to absolute path
        output_dir = self.config.output_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(_BENCHMARK_DIR, output_dir)
        output_path = save_results(metrics, output_dir)
        print(f"[{self.benchmark_id}] Results saved to {output_path}")

        return metrics

    @sync_to_async
    def cleanup(self):
        """Remove test data."""
        if not self.config.cleanup:
            print(f"[{self.benchmark_id}] Skipping cleanup (cleanup=False)")
            return

        print(f"[{self.benchmark_id}] Cleaning up...")

        # Delete test session and related data (need to handle FK constraints)
        if hasattr(self, 'test_session') and self.test_session:
            from data.models import Episode, Record
            try:
                # Delete records first (they reference episodes)
                episodes = Episode.objects.filter(session=self.test_session)
                Record.objects.filter(episode__in=episodes).delete()
                # Then delete episodes
                episodes.delete()
                # Then delete session
                self.test_session.delete()
            except Exception as e:
                print(f"[{self.benchmark_id}] Warning: Could not delete session: {e}")

        # Delete test participants and users
        for participant in self.test_participants:
            try:
                participant.delete()
            except Exception:
                pass

        for user in self.test_users:
            try:
                user.delete()
            except Exception:
                pass

        # Delete agents created for this benchmark
        if hasattr(self, 'agents'):
            for agent in self.agents:
                try:
                    agent.delete()
                except Exception:
                    pass

        print(f"[{self.benchmark_id}] Cleanup complete")

    async def execute(self) -> AggregateMetrics:
        """
        Full benchmark lifecycle: setup -> run -> cleanup.

        Returns:
            AggregateMetrics: Combined metrics from all participants
        """
        try:
            await self.setup()
            metrics = await self.run()
            return metrics
        finally:
            await self.cleanup()


async def run_benchmark(config: BenchmarkConfig) -> AggregateMetrics:
    """
    Run a single benchmark with the given configuration.

    Args:
        config: Benchmark configuration

    Returns:
        AggregateMetrics: Combined metrics from all participants
    """
    orchestrator = BenchmarkOrchestrator(config)
    return await orchestrator.execute()


async def run_scalability_suite(suite_config) -> List[AggregateMetrics]:
    """
    Run the full scalability suite.

    Args:
        suite_config: ScalabilitySuite configuration

    Returns:
        List[AggregateMetrics]: Results for each participant count
    """
    results = []

    for num_participants in suite_config.participant_counts:
        config = suite_config.get_config(num_participants)
        metrics = await run_benchmark(config)
        results.append(metrics)

    return results


class AIAgentOrchestrator:
    """
    Orchestrates AI agent benchmark: setup, execution, and cleanup.

    Unlike participant benchmarks, this creates AI agents with policies
    that run autonomously. One dummy participant is needed to trigger
    the session start.
    """

    def __init__(self, config: AIAgentConfig):
        self.config = config
        self.test_user: Optional[User] = None
        self.test_participant: Optional[Participant] = None
        self.test_session: Optional[Session] = None
        self.experiment: Optional[Experiment] = None
        self.room_id: str = ""
        self.benchmark_id: str = f"benchmark_ai_{config.num_agents}a_{uuid.uuid4().hex[:8]}"
        self.runner: Optional[Runner] = None
        self.agents: List[Agent] = []
        self.policy: Optional[Policy] = None

    async def setup(self) -> None:
        """Set up AI agent benchmark infrastructure."""
        print(f"[{self.benchmark_id}] Setting up AI agent benchmark...")

        if self.config.runner_connection_key:
            await self._verify_runner()
        else:
            print(f"[{self.benchmark_id}] Warning: No runner connection key provided.")

        await self._setup_environment()
        await self._setup_policy()
        await self._setup_experiment()
        await self._create_dummy_participant()
        await self._create_session()

        print(f"[{self.benchmark_id}] Setup complete")

    @sync_to_async
    def _verify_runner(self):
        """Verify that a runner with the given connection key exists and is idle."""
        try:
            runner = Runner.objects.get(connection_key=self.config.runner_connection_key)
        except Runner.DoesNotExist:
            raise RuntimeError(
                f"No runner found with connection key '{self.config.runner_connection_key}'."
            )
        if runner.status != 'idle':
            raise RuntimeError(
                f"Runner '{runner.connection_key}' is not idle (status: {runner.status})."
            )
        self.runner = runner
        print(f"[{self.benchmark_id}] Verified runner: {runner.connection_key}")

    @sync_to_async
    def _setup_environment(self):
        """Create no-op environment in database."""
        runner_env_path = os.path.join(
            os.path.dirname(__file__), '..', 'runner', 'noop_environment.py'
        )
        benchmark_env_path = os.path.join(
            os.path.dirname(__file__), 'noop_environment.py'
        )

        if not os.path.exists(runner_env_path):
            os.makedirs(os.path.dirname(runner_env_path), exist_ok=True)
            shutil.copy(benchmark_env_path, runner_env_path)

        env, created = Environment.objects.get_or_create(
            name="NoOp Benchmark Environment",
            defaults={
                "description": "Minimal environment for measuring infrastructure overhead",
                "filepaths": {"environment": "noop_environment.py"},
                "metadata": {"max_steps": self.config.num_steps},
            }
        )
        if not created:
            env.metadata = {"max_steps": self.config.num_steps}
            env.save()
        self.environment = env

    @sync_to_async
    def _setup_policy(self):
        """Create random policy in database and copy file to runner directory."""
        # Copy policy file to runner directory
        runner_policy_path = os.path.join(
            os.path.dirname(__file__), '..', 'runner', 'random_policy.py'
        )
        benchmark_policy_path = os.path.join(
            os.path.dirname(__file__), 'random_policy.py'
        )

        if not os.path.exists(runner_policy_path):
            shutil.copy(benchmark_policy_path, runner_policy_path)

        # Create or get policy
        policy, created = Policy.objects.get_or_create(
            name="Random Benchmark Policy",
            defaults={
                "description": "Random policy for AI agent benchmarking",
                "filepaths": {"policy": "random_policy.py"},
                "checkpoint_interval": 0,
            }
        )
        self.policy = policy

    @sync_to_async
    def _setup_experiment(self):
        """Create or get benchmark experiment with AI agents."""
        # Create AI agents with the random policy
        self.agents = []
        for i in range(self.config.num_agents):
            role = f"ai_agent_{i}"
            agent, created = Agent.objects.get_or_create(
                role=role,
                defaults={
                    "name": f"Benchmark AI Agent {i}",
                    "participant": False,  # AI agent, not participant-controlled
                    "policy": self.policy,
                    "inputs_type": "actions",
                    "keyboard_inputs": {"default": 0},
                }
            )
            # Ensure policy is attached
            if agent.policy_id != self.policy.id:
                agent.policy = self.policy
                agent.participant = False
                agent.save()
            self.agents.append(agent)

        # Create or get experiment
        experiment, created = Experiment.objects.get_or_create(
            link=AI_AGENT_EXPERIMENT["link"],
            defaults={
                "name": AI_AGENT_EXPERIMENT["name"],
                "short_description": AI_AGENT_EXPERIMENT["short_description"],
                "long_description": AI_AGENT_EXPERIMENT["long_description"],
                "environment": self.environment,
                "target_fps": AI_AGENT_EXPERIMENT["target_fps"],
                "wait_for_inputs": AI_AGENT_EXPERIMENT["wait_for_inputs"],
                "number_of_episodes": AI_AGENT_EXPERIMENT["number_of_episodes"],
            }
        )

        # Clear and add agents
        experiment.agents.clear()
        for agent in self.agents:
            experiment.agents.add(agent)

        self.experiment = experiment

    @sync_to_async
    def _create_dummy_participant(self):
        """Create a single dummy participant to trigger session start."""
        consent, _ = Consent.objects.get_or_create(
            name="Benchmark Consent",
            defaults={
                "explanation_text": "Automated benchmark testing consent.",
            }
        )

        username = f"benchmark_user_{self.benchmark_id}_dummy"

        self.test_user = User.objects.create_user(
            username=username,
            password="benchmark_password",
        )

        self.test_participant = Participant.objects.create(
            user=self.test_user,
            external_id=f"bench_{self.benchmark_id}_dummy",
            consent=consent,
        )

    @sync_to_async
    def _create_session(self):
        """Create session for benchmark."""
        self.room_id = f"room_{self.benchmark_id}"

        session = Session.objects.create(
            experiment=self.experiment,
            room=self.room_id,
            status="ready",
        )

        # Add the dummy participant
        session.participants.add(self.test_participant)
        session.save()

        self.test_session = session

    @sync_to_async
    def _get_session_cookie(self) -> Tuple[str, str]:
        """Get authenticated session cookie for dummy participant."""
        from django.contrib.sessions.backends.db import SessionStore

        session = SessionStore()
        session['_auth_user_id'] = str(self.test_user.id)
        session['_auth_user_backend'] = 'django.contrib.auth.backends.ModelBackend'
        session['_auth_user_hash'] = self.test_user.get_session_auth_hash()
        session['role'] = 'dummy_observer'
        session['session'] = self.test_session.id
        session.save()

        return str(self.test_participant.id), session.session_key

    async def run(self) -> AggregateMetrics:
        """Execute the AI agent benchmark."""
        print(f"[{self.benchmark_id}] Starting AI agent benchmark with {self.config.num_agents} agents...")

        # Get session cookie
        participant_id, session_cookie = await self._get_session_cookie()

        # Build WebSocket URL
        ws_url = (
            f"ws://{self.config.host}:{self.config.port}"
            f"/experiment/{self.experiment.link}/run/{self.room_id}"
        )

        # Create a single simulator (the dummy participant just observes)
        simulator = ParticipantSimulator(
            participant_id=participant_id,
            session_cookie=session_cookie,
            ws_url=ws_url,
            role='dummy_observer',
            num_steps=self.config.num_steps,
            action_interval=0.0,  # No action delay
            connection_timeout=self.config.runner_timeout,
            verbose=self.config.verbose,
            send_actions=False,  # AI agents run policies, no participant input needed
        )

        # Run the simulator (it just observes steps, no actions to send)
        participant_metrics = await simulator.run()

        # Aggregate metrics (single participant, but represents all AI agents)
        metrics = aggregate_metrics(
            [participant_metrics],
            benchmark_id=self.benchmark_id,
            target_steps=self.config.num_steps,
        )

        print(f"[{self.benchmark_id}] Benchmark complete")
        print(metrics.summary())

        output_dir = self.config.output_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(_BENCHMARK_DIR, output_dir)
        output_path = save_results(metrics, output_dir)
        print(f"[{self.benchmark_id}] Results saved to {output_path}")

        return metrics

    @sync_to_async
    def cleanup(self):
        """Remove test data."""
        if not self.config.cleanup:
            print(f"[{self.benchmark_id}] Skipping cleanup")
            return

        print(f"[{self.benchmark_id}] Cleaning up...")

        if hasattr(self, 'test_session') and self.test_session:
            from data.models import Episode, Record
            try:
                episodes = Episode.objects.filter(session=self.test_session)
                Record.objects.filter(episode__in=episodes).delete()
                episodes.delete()
                self.test_session.delete()
            except Exception as e:
                print(f"[{self.benchmark_id}] Warning: Could not delete session: {e}")

        if self.test_participant:
            try:
                self.test_participant.delete()
            except Exception:
                pass

        if self.test_user:
            try:
                self.test_user.delete()
            except Exception:
                pass

        for agent in self.agents:
            try:
                agent.delete()
            except Exception:
                pass

        print(f"[{self.benchmark_id}] Cleanup complete")

    async def execute(self) -> AggregateMetrics:
        """Full benchmark lifecycle: setup -> run -> cleanup."""
        try:
            await self.setup()
            metrics = await self.run()
            return metrics
        finally:
            await self.cleanup()


async def run_ai_agent_benchmark(config: AIAgentConfig) -> AggregateMetrics:
    """Run a single AI agent benchmark with the given configuration."""
    orchestrator = AIAgentOrchestrator(config)
    return await orchestrator.execute()


async def run_ai_agent_scalability_suite(suite_config) -> List[AggregateMetrics]:
    """Run the full AI agent scalability suite."""
    results = []

    for num_agents in suite_config.agent_counts:
        config = suite_config.get_config(num_agents)
        metrics = await run_ai_agent_benchmark(config)
        results.append(metrics)

    return results


async def run_network_latency_suite(suite_config: NetworkLatencySuite) -> List[AggregateMetrics]:
    """Run the network latency test suite."""
    from .config import NETWORK_PRESETS
    results = []

    for latency_preset in suite_config.latency_presets:
        config = suite_config.get_config(latency_preset)
        latency_ms = NETWORK_PRESETS.get(latency_preset, 0.0) * 1000
        print(f"\nRunning benchmark with {latency_preset} latency ({latency_ms:.1f}ms)...")
        metrics = await run_benchmark(config)
        results.append(metrics)

    return results


async def run_image_size_suite(suite_config: ImageSizeSuite) -> List[AggregateMetrics]:
    """Run the image size test suite."""
    from .config import IMAGE_SIZE_PRESETS
    results = []

    for image_size_preset in suite_config.image_size_presets:
        config = suite_config.get_config(image_size_preset)
        image_size = IMAGE_SIZE_PRESETS.get(image_size_preset, (64, 64, 3))
        h, w, _ = image_size
        print(f"\nRunning benchmark with {image_size_preset} image size ({h}x{w})...")
        metrics = await run_benchmark(config)
        results.append(metrics)

    return results