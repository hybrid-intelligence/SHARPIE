"""Unit tests for the runner's logic, with the connection to the server mocked where needed."""
import json
from unittest.mock import Mock

from websockets.sync.client import ClientConnection

from sharpie.runner.manage import receive_message


def test_receive_message_returns_the_batched_participant_actions():
    """The runner returns the actions the server batched for participant agents."""
    websocket = Mock(spec=ClientConnection)
    websocket.recv.return_value = json.dumps({'batch_actions': {'agent_0': 2}})
    agents_settings = {'agent_0': {'participant': True}}

    assert receive_message(websocket, agents_settings) == {'agent_0': 2}


def test_receive_message_does_not_wait_without_participant_agents():
    """Without participant agents, the runner doesn't wait for a message."""
    websocket = Mock(spec=ClientConnection)
    agents_settings = {'agent_0': {'participant': False}}

    assert receive_message(websocket, agents_settings) == {}
    websocket.recv.assert_not_called()
