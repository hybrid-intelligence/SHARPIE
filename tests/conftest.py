"""Shared pytest fixtures for SHARPIE's tests."""
from pathlib import Path

import pytest
from channels.testing import WebsocketCommunicator
from django.contrib.auth.models import User
from django.test import Client
from django.urls import reverse
from django.utils import timezone

from sharpie.benchmark import noop_environment
from sharpie.webserver.accounts.models import Participant
from sharpie.webserver.experiment.models import Agent, Environment, Experiment
from sharpie.webserver.runner.models import Runner
from sharpie.webserver.server.asgi import application

NOOP_ENVIRONMENT = Path(noop_environment.__file__)
EXPERIMENT_LINK = 'test'
ROOM = 'room1'


@pytest.fixture
def in_memory_channel_layer(settings):
    """Use Channels' in-memory channel layer instead of Redis."""
    settings.CHANNEL_LAYERS = {'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'}}


@pytest.fixture
def noop_environment():
    """Path to the benchmark's NoOp environment, which SHARPIE can run without any setup."""
    return NOOP_ENVIRONMENT


@pytest.fixture
def make_experiment(db, noop_environment):
    """Return a function that creates the test experiment with the given participant roles."""
    def make(participant_roles=('agent_0',)):
        environment = Environment.objects.create(
            name='NoOp', filepaths={'environment': str(noop_environment)})
        experiment = Experiment.objects.create(
            name='Test experiment', link=EXPERIMENT_LINK, environment=environment,
            short_description='Test', long_description='Test')
        for role in participant_roles:
            experiment.agents.add(Agent.objects.create(role=role, name=role, participant=True))
        return experiment
    return make


@pytest.fixture
def experiment(make_experiment):
    """An experiment with the NoOp environment and one participant role, agent_0."""
    return make_experiment()


@pytest.fixture
def make_participant(db):
    """Return a function that creates a participant who has given consent."""
    def make(username='participant'):
        user = User.objects.create_user(username, password='password')
        Participant.objects.create(user=user, agreed_at=timezone.now())
        return user
    return make


@pytest.fixture
def participant(make_participant):
    """A participant who has given consent."""
    return make_participant()


@pytest.fixture
def join_room():
    """Return a function that logs a participant in, joins a room through the configuration page
    and returns their session ID."""
    def join(participant, experiment, role='agent_0', room=ROOM):
        client = Client()
        client.force_login(participant)
        response = client.post(reverse('config', args=[experiment.link]),
                               {'room': room, 'role': role})
        assert response.status_code == 302
        return client.cookies['sessionid'].value
    return join


@pytest.fixture
def participant_session_id(join_room, participant, experiment):
    """The session ID of the participant after joining room1 as agent_0."""
    return join_room(participant, experiment)


@pytest.fixture
def room_path():
    """Path of room1's game page; the page opens its WebSocket on the same path."""
    return reverse('run', args=[EXPERIMENT_LINK, ROOM])


@pytest.fixture
def runner(db):
    """A runner registered with the server under a known connection key."""
    return Runner.objects.create(connection_key='runner-key')


@pytest.fixture
async def room_connection(room_path):
    """Return a function that creates a connection to room1's game room, as a participant
    (by session ID) or as a runner; all connections are closed after the test."""
    communicators = []

    def connection(session_id=None, runner=None):
        headers = []
        if session_id is not None:
            headers.append((b'cookie', f'sessionid={session_id}'.encode()))
        if runner is not None:
            headers.append((b'authorization', runner.connection_key.encode()))
        communicator = WebsocketCommunicator(application, room_path, headers=headers)
        communicators.append(communicator)
        return communicator

    yield connection
    for communicator in communicators:
        await communicator.disconnect()
