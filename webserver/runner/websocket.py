import json
import gzip

from asgiref.sync import async_to_sync
from django.db.models.functions import Now
from django.utils import timezone
from channels.generic.websocket import WebsocketConsumer

from .models import Runner
from data.models import Session



class ConnectionConsumer(WebsocketConsumer):
    def connect(self):
        # Initiate variables
        self.runner = None
        self.scope['connection_key'] = None
        # Check if the authorization is available in the headers
        for h in self.scope['headers']:
            if h[0] == b'authorization':
                self.scope['connection_key'] = h[1].decode('utf-8')
                break
        
        if not self.scope['connection_key']:
            self.close(code=1008)  # Missing connection key
            return
        
        # Try to get a disconnected runner for this experiment, else create a new one
        try:
            self.runner = Runner.objects.get(connection_key=self.scope['connection_key'])
            self.runner.status = 'idle'
            self.runner.last_active = Now()
            self.runner.ip_address = self.scope['client'][0]
            self.runner.save()
        except Runner.DoesNotExist:
            self.close(code=1008)  # Incorrect connection key
            return

        self.accept()

    def receive(self, text_data=None, bytes_data=None):
        message = json.loads(text_data)
        if 'status' in message.keys() and message['status'] == 'idle':
            self.runner.status = 'idle'
            self.runner.last_active = Now()
            self.runner.ip_address = self.scope['client'][0]
            self.runner.save()
            # Check if there are any waiting queues for this experiment
            try:
                # Get the first pending session
                session = Session.objects.filter(status='pending').first()
                if session:
                    # Updating session status
                    session.status = 'running'
                    session.save()
                    # Updating runner status
                    self.runner.status = 'running'
                    self.runner.session = session
                    self.runner.last_active = Now()
                    self.runner.save()
                    # Send message to runner to start the episode
                    self.send(json.dumps({"experiment": session.experiment.link, "room": session.room}))
                else:
                    # Send message to runner that there are no session in the queue
                    self.send(json.dumps({"message": "No session in the queue"}))
            except Session.DoesNotExist:
                # Send message to runner that there are no session in the queue
                self.send(json.dumps({"message": "No session in the queue"}))

    def disconnect(self, close_code):
        if self.runner:
            self.runner.status = 'disconnected'
            self.runner.last_active = Now()
            self.runner.ip_address = self.scope['client'][0]
            self.runner.save()