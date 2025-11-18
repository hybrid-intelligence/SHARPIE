import json
import gzip

from asgiref.sync import async_to_sync
from django.db.models.functions import Now
from django.utils import timezone
from channels.generic.websocket import WebsocketConsumer

from .models import Experiment, Trial, Interaction, Runner, Queue
from mysite.settings import REDIRECT_AFTER_EXPERIMENT






class InterfaceConsumer(WebsocketConsumer):
    def connect(self):
        # Get the experiment link from the URL
        self.link = self.scope["url_route"]["kwargs"]["link"]

        # Get the room name, if not defined in session that is a runner
        if 'room_name' not in self.scope['session'].keys():
            self.room_name = 'runner'
            self.trial = None

            self.accept()
            # Join room group
            async_to_sync(self.channel_layer.group_add)(
                self.room_name + self.link, self.channel_name
            )
        else:
            self.room_name = self.scope['session']['room_name']
            self.trial = Trial(
                experiment=Experiment.objects.get(link=self.link),
                room_name=self.room_name,
                user=self.scope['user'] if self.scope['user'].is_authenticated else None,
                agent_played=self.scope['session']['agent']
            )

            self.queue = Queue.objects.filter(experiment=self.trial.experiment, room_name=self.room_name, status__in=['waiting', 'running']).first()
            if self.queue and self.queue.created_at < timezone.now() - timezone.timedelta(minutes=10):  # If the queue was created more than 10 min ago, we consider it dead
                self.queue.status = 'dead'
                self.queue.save()
            if not self.queue:
                self.queue = Queue(experiment=self.trial.experiment, room_name=self.room_name, users_waiting=0)
                self.queue.status = 'waiting'
            
            if self.queue.users_waiting + 1 > self.trial.experiment.users_needed:
                self.accept()
                self.send(json.dumps({"message": "A experiment already started in this room, please select another one."}))
                self.close()
            else:
                self.queue.users_waiting += 1
                self.queue.save()
                self.trial.save()

                self.accept()
                # Join room group
                async_to_sync(self.channel_layer.group_add)(
                    self.room_name + self.link, self.channel_name
                )
    
    # Create message for the runner
    def runner_message(self, key, value):
        runner_message = {"type": "websocket.message", "room": None, "received": False, "actions": [], "fps": 0, "session": None, "users_needed": 1}
        runner_message[key] = value
        runner_message['session'] = dict(self.scope['session'])
        runner_message['room'] = self.room_name
        return runner_message

    # Send message
    def websocket_message(self, event):
        # Forward message to the browser WebSocket
        self.send(json.dumps(event))
        # Save interaction to the database
        if self.trial:
            interaction = Interaction(
                trial=self.trial,
                step=event['step'],
                observations=event['observations'],
                actions=event['actions'],
                rewards=event['rewards']
            )
            interaction.save()

    # Receive message from WebSocket
    def receive(self, text_data=None, bytes_data=None):
        # If it is text data, that comes from the browser
        if(text_data):
            message = json.loads(text_data)
            # Send message to runner
            async_to_sync(self.channel_layer.group_send)(
                'runner' + self.link, self.runner_message("actions", message['actions'])
            )
        # Else that comes from the runner
        elif(bytes_data):
            message = json.loads(gzip.decompress(bytes_data))
            message['type'] = "websocket.message"
            if message['terminated']:
                episodes_to_complete = Experiment.objects.get(link=self.link).episodes_to_complete
                n_trials = Trial.objects.filter(experiment__link=self.link, room_name=message['room']).count()
                if n_trials >= episodes_to_complete:
                    message['redirect'] = REDIRECT_AFTER_EXPERIMENT
                else:
                    message['redirect'] = None
            # Send message to room group
            async_to_sync(self.channel_layer.group_send)(
                message['room'] + self.link, message
            )

    def disconnect(self, close_code):
        # Send message to all connected users that this user has disconnected
        message = {
            "type": "websocket.message",
            "message": "A user has disconnected",
        }
        async_to_sync(self.channel_layer.group_send)(
            self.room_name + self.link, message
        )
        async_to_sync(self.channel_layer.group_send)(
            'runner' + self.link, message
        )
        # Leave room group
        async_to_sync(self.channel_layer.group_discard)(
            self.room_name + self.link, self.channel_name
        )
        # If the episode has ended, we save the end time of the trial
        if self.trial:
            self.trial.ended_at = Now()
            self.trial.save() 
            self.queue.status = 'terminated'
            self.queue.save()








class QueueConsumer(WebsocketConsumer):
    def connect(self):
        # Try to get a disconnected runner for this experiment, else create a new one
        try:
            self.runner = Runner.objects.get(ip_address=self.scope['client'][0])
            self.runner.status = 'idle'
            self.runner.current_room = None
            self.runner.last_active = Now()
            self.runner.experiment = None
            self.runner.save()
        except Runner.DoesNotExist:
            self.runner = Runner(status='idle', ip_address=self.scope['client'][0])
            self.runner.save()

        self.accept()

    def receive(self, text_data=None, bytes_data=None):
        message = json.loads(text_data)
        if 'status' in message.keys() and message['status'] == 'idle':
            self.runner.status = 'idle'
            self.runner.current_room = None
            self.runner.last_active = Now()
            self.runner.experiment = None
            self.runner.save()
            # Check if there are any waiting queues for this experiment
            try:
                queue = Queue.objects.filter(status='waiting').first()
                if queue and queue.users_waiting >= queue.experiment.users_needed:
                    queue.status = 'running'
                    queue.save()
                    self.runner.status = 'running'
                    self.runner.current_room = queue.room_name
                    self.runner.last_active = Now()
                    self.runner.experiment = queue.experiment
                    self.runner.save()
                    # Send message to runner to start the episode
                    self.send(json.dumps({"experiment": queue.experiment.link, "room": queue.room_name, "users_needed": queue.users_waiting}))
                else:
                    # Send message to runner that there are no experiments in the queue
                    self.send(json.dumps({"message": "No experiments in the queue"}))
            except Queue.DoesNotExist:
                # Send message to runner that there are no experiments in the queue
                self.send(json.dumps({"message": "No experiments in the queue"}))

    def disconnect(self, close_code):
        self.runner.status = 'disconnected'
        self.runner.current_room = None
        self.runner.last_active = Now()
        self.runner.save()