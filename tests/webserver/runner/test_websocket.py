"""Tests for the runner waiting room (ConnectionConsumer)."""
import pytest
from asgiref.sync import sync_to_async
from channels.testing import WebsocketCommunicator

from sharpie.webserver.data.models import Session
from sharpie.webserver.server.asgi import application

pytestmark = [pytest.mark.django_db(transaction=True),
              pytest.mark.usefixtures('in_memory_channel_layer')]

RUNNER_ADDRESS = ('127.0.0.1', 50000)


def waiting_room_connection(key=None):
    """A connection to the waiting room, with the client address Daphne would provide."""
    headers = [(b'authorization', key.encode())] if key else []
    communicator = WebsocketCommunicator(application, '/runner/connection', headers=headers)
    communicator.scope['client'] = RUNNER_ADDRESS
    return communicator


@pytest.fixture
def waiting_game(experiment):
    """A game session waiting for a runner."""
    return Session.objects.create(experiment=experiment, room='room1', status='pending')


async def test_unknown_runner_key_is_refused(runner):
    """A runner connecting with a key the server doesn't know is refused."""
    connected, close_code = await waiting_room_connection('wrong-key').connect()

    assert not connected
    assert close_code == 1008


async def test_runner_without_a_key_is_refused():
    """A runner connecting without any key is refused."""
    connected, close_code = await waiting_room_connection().connect()

    assert not connected
    assert close_code == 1008


async def test_runner_with_a_known_key_is_accepted_and_marked_idle(runner):
    """A registered runner is let in, marked idle, and its address is recorded."""
    communicator = waiting_room_connection(runner.connection_key)

    connected, _ = await communicator.connect()

    assert connected
    await sync_to_async(runner.refresh_from_db)()
    assert runner.status == 'idle'
    assert runner.ip_address == '127.0.0.1'
    await communicator.disconnect()


async def test_idle_runner_is_handed_a_waiting_game(runner, waiting_game):
    """An idle runner is told which game to run, and the game is marked running."""
    communicator = waiting_room_connection(runner.connection_key)
    await communicator.connect()

    await communicator.send_json_to({'status': 'idle'})
    message = await communicator.receive_json_from()

    assert message == {'experiment': waiting_game.experiment.link, 'room': waiting_game.room}
    await sync_to_async(waiting_game.refresh_from_db)()
    assert waiting_game.status == 'running'
    await communicator.disconnect()


async def test_idle_runner_is_told_when_no_game_is_waiting(runner):
    """An idle runner is told there's no game when none is waiting."""
    communicator = waiting_room_connection(runner.connection_key)
    await communicator.connect()

    await communicator.send_json_to({'status': 'idle'})

    assert await communicator.receive_json_from() == {'message': 'No session in the queue'}
    await communicator.disconnect()


async def test_runner_that_leaves_is_marked_disconnected(runner):
    """A runner that leaves the waiting room is marked disconnected."""
    communicator = waiting_room_connection(runner.connection_key)
    await communicator.connect()

    await communicator.disconnect()

    await sync_to_async(runner.refresh_from_db)()
    assert runner.status == 'disconnected'
