"""Tests for the game room (RunConsumer) with SHARPIE's Redis channel layer, on a real Redis."""
import copy

import pytest

pytestmark = [pytest.mark.redis, pytest.mark.django_db(transaction=True)]


@pytest.fixture
def redis_channel_layer(settings, redis_proc, redisdb):
    """Use SHARPIE's own channel layer settings, pointed at the test's Redis server."""
    channel_layers = copy.deepcopy(settings.CHANNEL_LAYERS)
    channel_layers['default']['CONFIG']['hosts'] = [(redis_proc.host, redis_proc.port)]
    settings.CHANNEL_LAYERS = channel_layers


@pytest.fixture
def two_participants_session_ids(make_experiment, make_participant, join_room):
    """Two participants who joined room1 as agent_0 and agent_1; returns their session IDs."""
    experiment = make_experiment(participant_roles=('agent_0', 'agent_1'))
    return (join_room(make_participant('alice'), experiment, role='agent_0'),
            join_room(make_participant('bob'), experiment, role='agent_1'))


async def test_runner_frame_reaches_both_participants(redis_channel_layer, room_connection, runner,
                                                      two_participants_session_ids):
    """A frame the runner broadcasts reaches every participant in the room, through Redis."""
    alice_id, bob_id = two_participants_session_ids
    alice = room_connection(session_id=alice_id)
    bob = room_connection(session_id=bob_id)
    runner_ws = room_connection(runner=runner)
    for communicator in (alice, bob, runner_ws):
        connected, _ = await communicator.connect()
        assert connected

    await runner_ws.send_json_to({'type': 'broadcast', 'image': 'frame-1'})

    assert (await alice.receive_json_from())['image'] == 'frame-1'
    assert (await bob.receive_json_from())['image'] == 'frame-1'
