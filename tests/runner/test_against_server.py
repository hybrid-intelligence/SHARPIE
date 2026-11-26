"""Tests for the runner program against a real WebSocket server that plays SHARPIE's side."""
import json
import subprocess
import sys
import threading

import pytest
from websockets.sync.server import serve

from sharpie.runner.manage import start_experiment

# The WebSocket close code with which the server dies in the reconnect test: RFC 6455
# §7.4.1 "Internal Error". websockets raises ConnectionClosedError for it, unlike the
# graceful 1000/1001 closes, and only the error variant reconnects the runner.
SERVER_ERROR_CLOSE_CODE = 1011


@pytest.fixture
def start_server():
    """Return a function that starts a WebSocket server on a free port with the given handler;
    all servers are stopped after the test."""
    servers = []

    def start(handler):
        server = serve(handler, 'localhost', 0)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        servers.append((server, thread))
        return server.socket.getsockname()[1]

    yield start
    for server, thread in servers:
        server.shutdown()
        thread.join()


@pytest.fixture
def start_game_room(start_server, noop_environment):
    """Return a function that starts a server playing the game room for one episode."""
    def start(max_steps=1, agents=None, batch_actions=None):
        """Start the game room with these settings and, if given, these participant actions."""
        received = {'frames': []}
        finished = threading.Event()

        def game_room(websocket):
            received['path'] = websocket.request.path
            received['key'] = websocket.request.headers['authorization']
            received['request'] = json.loads(websocket.recv())
            websocket.send(json.dumps({
                'files': {'environment': {'path': str(noop_environment), 'content': ''}},
                'metadata': {'max_steps': max_steps}}))
            websocket.send(json.dumps(agents or {}))
            websocket.send(json.dumps({'target_fps': 1000, 'wait_for_inputs': False}))
            for message in websocket:
                frame = json.loads(message)
                received['frames'].append(frame)
                if batch_actions is not None and not (frame['terminated'] or frame['truncated']):
                    websocket.send(json.dumps({'batch_actions': batch_actions}))
            finished.set()

        return start_server(game_room), received, finished
    return start


@pytest.fixture
def start_runner():
    """Return a function that starts the runner program against a server port;
    all runner programs are stopped after the test."""
    processes = []

    def start(port, key='runner-key'):
        process = subprocess.Popen([sys.executable, '-m', 'sharpie.runner.manage', 'runserver',
                                    '--port', str(port), '--connection-key', key])
        processes.append(process)
        return process

    yield start
    for process in processes:
        process.terminate()
        process.wait()


def test_runner_plays_an_episode_in_the_game_room(start_game_room):
    """The runner connects to the room with its key, asks for settings and sends every frame."""
    port, received, finished = start_game_room(max_steps=2)

    start_experiment('localhost', port, 'runner-key', 'test', 'room1')

    assert finished.wait(timeout=5)
    assert received['path'] == '/experiment/test/run/room1'
    assert received['key'] == 'runner-key'
    assert received['request'] == {'type': 'private', 'message': 'settings'}
    assert [frame['step'] for frame in received['frames']] == [1, 2, 3]
    assert received['frames'][-1]['truncated']


def test_runner_sends_one_frame_per_step(start_game_room):
    """The runner sends a frame for every step, plus the final state."""
    port, received, finished = start_game_room(max_steps=5)

    start_experiment('localhost', port, 'runner-key', 'test', 'room1')

    assert finished.wait(timeout=5)
    assert [frame['step'] for frame in received['frames']] == [1, 2, 3, 4, 5, 6]


def test_runner_uses_the_participants_actions(start_game_room):
    """With a participant agent, the runner applies the actions the room sends back."""
    agents = {'agent_0': {'participant': True, 'inputs_type': 'actions',
                          'keyboard_inputs': {'default': 0}}}
    port, received, finished = start_game_room(agents=agents, batch_actions={'agent_0': 1})

    start_experiment('localhost', port, 'runner-key', 'test', 'room1')

    assert finished.wait(timeout=5)
    assert received['frames'][1]['actions'] == {'agent_0': 1}


def test_runner_reconnects_when_the_connection_drops(start_server, start_runner):
    """When the server drops the connection, the runner program connects again."""
    paths = []
    reconnected = threading.Event()

    def waiting_room(websocket):
        paths.append(websocket.request.path)
        websocket.recv()
        if len(paths) == 1:
            websocket.close(code=SERVER_ERROR_CLOSE_CODE)
        else:
            reconnected.set()
            websocket.send(json.dumps({'message': 'No session in the queue'}))

    start_runner(start_server(waiting_room))

    assert reconnected.wait(timeout=10)
    assert paths == ['/runner/connection', '/runner/connection']
