"""Tests for the game room (RunConsumer)."""
import pytest

pytestmark = [pytest.mark.django_db(transaction=True),
              pytest.mark.usefixtures('in_memory_channel_layer')]


async def test_runner_frame_reaches_the_participant(
        room_connection, runner, participant_session_id):
    """A frame the runner broadcasts reaches the participant in the same room."""
    participant_ws = room_connection(session_id=participant_session_id)
    runner_ws = room_connection(runner=runner)
    connected, _ = await participant_ws.connect()
    assert connected
    connected, _ = await runner_ws.connect()
    assert connected

    await runner_ws.send_json_to({'type': 'broadcast', 'image': 'frame-1'})
    message = await participant_ws.receive_json_from()

    assert message['image'] == 'frame-1'
