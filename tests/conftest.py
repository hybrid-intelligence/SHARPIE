"""Shared pytest fixtures for SHARPIE's tests."""
from pathlib import Path

import pytest
from django.contrib.auth.models import User
from django.utils import timezone

from sharpie.benchmark import noop_environment
from sharpie.webserver.accounts.models import Participant
from sharpie.webserver.experiment.models import Agent, Environment, Experiment

NOOP_ENVIRONMENT = Path(noop_environment.__file__)
EXPERIMENT_LINK = 'test'


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
