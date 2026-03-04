import json

from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.db.models.functions import Now

from accounts.models import Participant
from runner.models import Runner
from data.models import Session, Episode, Record
from .consumer_helpers import RunConsumerHelpers, decode_data


# Store runner channel names per room for direct messaging
# Key: (link, room), Value: channel_name
_runner_channels = {}

# Action batching: collect actions and send as a single message
# Key: (link, room), Value: {"actions": {role: action, ...}, "count": N, "expected": M}
_action_buffers = {}


class RunConsumer(RunConsumerHelpers, AsyncWebsocketConsumer):
    """WebSocket consumer for experiment execution.

    Handles communication between participants, runners, and the experiment.
    Participants send actions; runners broadcast state updates.
    """

    async def verify_runner(self):
        """Verify if the connection is from an authenticated runner."""
        connection_key = None
        # Check if the authorization is available in the headers
        for h in self.scope['headers']:
            if h[0] == b'authorization':
                connection_key = h[1].decode('utf-8')
                break

        if not connection_key:
            # Missing connection key
            return None

        # Try to get the runner from the connection key
        try:
            runner = await database_sync_to_async(Runner.objects.get)(connection_key=connection_key)
            return runner
        except Runner.DoesNotExist:
            # Incorrect connection key
            await self.close(code=1008)

        return None

    async def connect(self):
        """Handle WebSocket connection."""
        # Get the experiment link and room from the URL
        self.link = self.scope["url_route"]["kwargs"]["link"]
        self.room = self.scope["url_route"]["kwargs"]["room"]
        # Check if it is the runner or not
        self.runner = await self.verify_runner()
        # If it is not a runner, it should be an authenticated participant
        if not self.runner and not self.scope['user'].is_authenticated:
            await self.close(code=1003)
            return
        # Retrieve the correct sessions and episode
        self.session = await database_sync_to_async(
            Session.objects.get
        )(experiment__link=self.link, room=self.room, status__in=['not_ready', 'ready', 'pending', 'running'])
        # Retrieve the correct episode (handle race condition with get_or_create)
        self.episode, _ = await database_sync_to_async(
            Episode.objects.get_or_create
        )(session=self.session, ended_at__isnull=True)
        if self.runner:
            self.role = None
        # If it is not a runner, we can retrieve the participant role
        else:
            self.role = self.scope["session"]["role"]
            # Atomically increment connected_participants and check if all connected
            all_connected = await database_sync_to_async(self._do_increment)()
            if all_connected:
                self.session.status = 'pending'
                await database_sync_to_async(self.session.save)(update_fields=['status'])
        # Initialize in-memory record storage (only for runner)
        if self.runner:
            self.records_buffer = []
        else:
            self.records_buffer = None

        await self.accept()
        # Join room group
        await self.channel_layer.group_add(
            f"{self.link}_{self.room}",
            self.channel_name
        )
        # Store runner channel for direct messaging
        if self.runner:
            _runner_channels[(self.link, self.room)] = self.channel_name
            # Initialize action buffer with expected participant count
            num_participants = await database_sync_to_async(
                self.session.participants.count
            )()
            _action_buffers[(self.link, self.room)] = {
                "actions": {},
                "expected": num_participants
            }

    async def websocket_message(self, event):
        """Forward WebSocket message to client."""
        if event['from'] != self.channel_name:
            await self.send(json.dumps(event))

    async def receive(self, text_data=None):
        """Handle incoming WebSocket message."""
        message = json.loads(text_data)
        if message["type"] == 'broadcast':
            await self._handle_broadcast(message)
        elif message.get("type") == 'private' and message.get("message") == 'action':
            await self._handle_action(message)
        elif message.get("message") == 'settings':
            await self._handle_settings()

    async def _handle_broadcast(self, message):
        """Handle broadcast message from runner."""
        # Update the step count and store record in memory
        if "step" in message:
            # Store in memory instead of creating it immediately
            self.episode.duration_steps = message["step"]
            self.records_buffer.append(
                Record(
                    episode=self.episode,
                    step_index=message["step"],
                    state=decode_data(message["observations"]),
                    action=decode_data(message["actions"]),
                    reward=decode_data(message["rewards"])
                )
            )
        # Update the episode and session status
        if "terminated" in message and (message["terminated"] or message["truncated"]):
            self.episode.completed = True
            await database_sync_to_async(self.episode.save)()
            # Check if all episodes completed - need to query database
            episode_info = await database_sync_to_async(self._fetch_episode_info)()
            if episode_info['completed_count'] >= episode_info['number_of_episodes']:
                message["completed"] = True
                if episode_info['redirect_url']:
                    message["redirect"] = episode_info['redirect_url']
        # Forward message to participants
        message["type"] = "websocket.message"
        message["from"] = self.channel_name
        if self.role:
            message["role"] = self.role
        # For now, we don't forward extra info to participants
        message["observations"] = []
        message["actions"] = []
        message["rewards"] = []
        await self.channel_layer.group_send(
            f"{self.link}_{self.room}",
            message
        )

    async def _handle_action(self, message):
        """Handle action message from participant with batching."""
        # Only participants should send actions, not the runner
        if not self.runner and self.role:
            # Get or create action buffer for this room
            room_key = (self.link, self.room)
            buffer = _action_buffers.get(room_key)

            if buffer is None:
                # Initialize buffer if not exists (fallback)
                num_participants = await database_sync_to_async(
                    self.session.participants.count
                )()
                buffer = {"actions": {}, "expected": num_participants}
                _action_buffers[room_key] = buffer

            # Add action to buffer
            buffer["actions"][self.role] = message.get("action")

            # Check if we have all expected actions
            if len(buffer["actions"]) >= buffer["expected"]:
                # Send batched actions to runner
                runner_channel = _runner_channels.get(room_key)
                if runner_channel:
                    batch_message = {
                        "type": "websocket.message",
                        "from": self.channel_name,
                        "batch_actions": buffer["actions"],
                    }
                    await self.channel_layer.send(
                        runner_channel,
                        batch_message
                    )
                # Clear buffer for next round
                buffer["actions"] = {}

    async def _handle_settings(self):
        """Handle settings request from runner."""
        # Fetch all needed data from database synchronously
        settings_data = await database_sync_to_async(self._fetch_settings_data)()

        # Send environment settings
        await self.send(json.dumps(settings_data['environment']))

        # Send agents settings
        await self.send(json.dumps(settings_data['agents']))

        # Send experiment settings
        await self.send(json.dumps(settings_data['experiment']))

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        # Check if session exists (may have been deleted by cleanup or error)
        if not hasattr(self, 'session') or self.session is None:
            return

        # If the episode has ended, we free up the runner instance
        if self.runner:
            # Remove runner channel from registry
            _runner_channels.pop((self.link, self.room), None)
            # Remove action buffer
            _action_buffers.pop((self.link, self.room), None)
            try:
                self.runner.status = 'idle'
                self.runner.session = None
                await database_sync_to_async(self.runner.save)()
            except Exception:
                pass  # Runner may have been deleted

            # Update the episode
            if hasattr(self, 'episode') and self.episode:
                try:
                    self.episode.ended_at = Now()
                    await database_sync_to_async(self.episode.save)()
                except Exception:
                    pass  # Episode may have been deleted or session FK missing

            # Bulk create all records from the buffer
            if self.records_buffer:
                try:
                    await database_sync_to_async(Record.objects.bulk_create)(self.records_buffer)
                except Exception:
                    pass  # Records may fail if episode was deleted

            # Reset the session - use sync helper
            try:
                await database_sync_to_async(self._do_session_update)()
            except Exception:
                # Session may have been deleted
                return

        # If this is a participant (not a runner) and the session is completed or aborted,
        # we need to update their session to reflect they're no longer in that session
        if not self.runner and hasattr(self, 'session'):
            # Refresh the session from the database to get the latest status
            try:
                await database_sync_to_async(self.session.refresh_from_db)()
            except Exception:
                # Session may have been deleted
                pass
            else:
                # Check if session status is completed or aborted
                if self.session.status in ['completed', 'aborted']:
                    # Update the user's session to indicate they're no longer in this session
                    self.scope["session"]["session"] = None
                    try:
                        await database_sync_to_async(self.scope["session"].save)()
                    except Exception:
                        pass  # Session may have been deleted

        # Leave room group
        await self.channel_layer.group_discard(
            f"{self.link}_{self.room}",
            self.channel_name
        )
        # Broadcast disconnect message
        message = {
            "type": "websocket.message",
            "from": self.channel_name,
            "error": "A user has disconnected"
        }
        await self.channel_layer.group_send(
            f"{self.link}_{self.room}",
            message
        )