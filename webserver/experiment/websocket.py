import json
import os

from asgiref.sync import async_to_sync
from django.db.models.functions import Now
from django.utils import timezone
from channels.generic.websocket import WebsocketConsumer

from accounts.models import Participant
from runner.models import Runner
from data.models import Session, Episode, Record




class RunConsumer(WebsocketConsumer):
    def verify_runner(self):
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
            runner = Runner.objects.get(connection_key=connection_key)
            return runner
        except Runner.DoesNotExist:
            # Incorrect connection key
            self.close(code=1008)

        return None

    def connect(self):
        # Get the experiment link and room from the URL
        self.link = self.scope["url_route"]["kwargs"]["link"]
        self.room = self.scope["url_route"]["kwargs"]["room"]
        # Check if it is the runner or not
        self.runner = self.verify_runner()
        # If it is not a runner, it should be an authenticated participant
        if not self.runner and not self.scope['user'].is_authenticated:
            self.close(code=1003)
        # Retrieve the correct sessions and episode
        self.session = Session.objects.get(experiment__link=self.link, room=self.room, status__in=['not_ready', 'ready', 'pending', 'running'])
        # Retrieve the correct episode
        try:
            self.episode = Episode.objects.get(session=self.session, ended_at__isnull=True)
        except Episode.DoesNotExist:
            # Create a new episode
            self.episode = Episode.objects.create(session=self.session)
        if self.runner:
            self.role = None
        # If it is not a runner, we can retrieve the participant role
        else:
            self.role = self.scope["session"]["role"]
            self.session.connected_participants += 1
            # If all participants connected, we can set the session status to 'pending'
            if self.session.connected_participants == len(self.session.participants.all()):
                self.session.status = 'pending'
            self.session.save()
        # Initialize in-memory record storage (only for runner)
        if self.runner:
            self.records_buffer = []
        else:
            self.records_buffer = None

        self.accept()
        # Join room group
        async_to_sync(self.channel_layer.group_add)(
            f"{self.link}_{self.room}", self.channel_name
        )




    # Send message
    def websocket_message(self, event):
        # Forward message
        if(event['from'] != self.channel_name):
            self.send(json.dumps(event))

    # Receive message from WebSocket and forward it to group
    def receive(self, text_data=None):
        message = json.loads(text_data)
        if message["type"] == 'broadcast':
            # Update the step count and store record in memory
            if "step" in message:
                # Store in memory instead of creating it immediately
                self.episode.duration_steps = message["step"]
                self.records_buffer.append(
                    Record(
                        episode=self.episode,
                        step_index=message["step"],
                        state=message["observations"],
                        action=message["actions"],
                        reward=message["rewards"]
                    )
                )
            # Update the episode and session status
            if "terminated" in message and (message["terminated"] or message["truncated"]):
                self.episode.completed = True
                self.episode.save()
                # If all episodes have been completed
                if len(self.session.episodes.filter(completed=True)) >= self.session.experiment.number_of_episodes:
                    message["completed"] = True
                    if self.session.experiment.redirect_url:
                        message["redirect"] = self.session.experiment.redirect_url
            # Forward message to participants
            message["type"] = "websocket.message"
            message["from"] = self.channel_name
            if self.role:
                message["role"] = self.role
            async_to_sync(self.channel_layer.group_send)(
                f"{self.link}_{self.room}", message
            )

        elif message["message"] == 'settings':
            # Send environment settings
            message = {'files': {}}
            for name, filepath in self.session.experiment.environment.filepaths.items():
                # Try to find the file and last modification date
                #modification_time = os.path.getmtime(os.path.join('..','runner',filepath))
                # Send file if it is not running on the same machine
                if self.runner.ip_address != '127.0.0.1':
                    with open(os.path.join('..','runner',filepath), 'r') as file:
                        code = file.read()   
                    message['files'][name] = {'path': filepath, 'content': code}
                else:
                    message['files'][name] = {'path': filepath, 'content': None}
            self.send(json.dumps(message))
            # Send agents settings
            message = {}
            for agent in self.session.experiment.agents.all():
                message[agent.role] = {'participant': agent.participant,
                                       'keyboard_inputs': agent.keyboard_inputs,
                                       'multiple_keyboard_inputs': agent.multiple_keyboard_inputs,
                                       'textual_inputs': agent.textual_inputs,
                                       'inputs_type': agent.inputs_type}
                if agent.policy:
                    message[agent.role]['policy'] = {'checkpoint_interval': agent.policy.checkpoint_interval}
                    message[agent.role]['policy']['files'] = {}
                    for name, filepath in agent.policy.filepaths.items():
                        # Try to find the file and last modification date
                        #modification_time = os.path.getmtime(os.path.join('..','runner',filepath))
                        # Send file if it is not running on the same machine
                        if self.runner.ip_address != '127.0.0.1':
                            with open(os.path.join('..','runner',filepath), 'r') as file:
                                code = file.read()   
                            message[agent.role]['policy']['files'][name] = {'path': filepath, 'content': code}
                        else:
                            message[agent.role]['policy']['files'][name] = {'path': filepath, 'content': None}
            self.send(json.dumps(message))
            # Send experiment settings
            message = {'conda_environment': self.session.experiment.conda_environment,
                       'target_fps': self.session.experiment.target_fps,
                       'wait_for_inputs': self.session.experiment.wait_for_inputs}
            self.send(json.dumps(message))



    def disconnect(self, close_code):
        # If the episode has ended, we free up the runner instance
        if self.runner:
            self.runner.status = 'idle'
            self.runner.session = None
            self.runner.save()
            # Update the episode
            self.episode.ended_at = Now()
            self.episode.save()
            # Bulk create all records from the buffer
            if self.records_buffer:
                Record.objects.bulk_create(self.records_buffer)
            # Reset the session
            self.session.refresh_from_db()
            self.session.connected_participants = 0
            # If all episodes have been completed
            if len(self.session.episodes.filter(completed=True)) >= self.session.experiment.number_of_episodes:
                self.session.status = 'completed'
                self.session.end_time = Now()
            # If some episodes have been aborted
            elif len(self.session.episodes.all()) >= self.session.experiment.number_of_episodes:
                self.session.status = 'aborted'
                self.session.end_time = Now()
            # Otherwise, put the session status as ready
            else:
                self.session.status = 'ready'
            self.session.save()

        # If this is a participant (not a runner) and the session is completed or aborted,
        # we need to update their session to reflect they're no longer in that session
        if not self.runner and hasattr(self, 'session'):
            # Refresh the session from the database to get the latest status
            self.session.refresh_from_db()
            # Check if session status is completed or aborted
            if self.session.status in ['completed', 'aborted']:
                # Update the user's session to indicate they're no longer in this session
                self.scope["session"]["session"] = None
                self.scope["session"].save()

        # Leave room group
        async_to_sync(self.channel_layer.group_discard)(
            f"{self.link}_{self.room}", self.channel_name
        )
        # Broadcast disconnect message
        message = {}
        message["type"] = "websocket.message"
        message["from"] = self.channel_name
        message["error"] = "A user has disconnected"
        async_to_sync(self.channel_layer.group_send)(
            f"{self.link}_{self.room}", message
        )