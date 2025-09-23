import json
import gzip

from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer






# Websocker consumer that inherits from the consumer template
class Consumer(WebsocketConsumer):
    
    def connect(self):
        # Get the room name, if not defined in session that is a runner
        if 'room_name' not in self.scope['session'].keys():
            self.room_name = 'runner'
        else:
            self.room_name = self.scope['session']['room_name']
        
        # Join room group
        async_to_sync(self.channel_layer.group_add)(
            self.room_name, self.channel_name
        )
        
        self.accept()

        # If it is not a runner, start the experiment
        if self.room_name != 'runner':
            # Send message to runner to start the experiment
            async_to_sync(self.channel_layer.group_send)(
                'runner', self.runner_message("room", self.room_name)
            )
        





    # Create message for the runner
    def runner_message(self, key, value):
        runner_message = {"type": "websocket.message", "room": None, "received": False, "actions": [], "fps": 0, "session": None}
        runner_message[key] = value
        runner_message['session'] = dict(self.scope['session'])
        return runner_message

    # Send message
    def websocket_message(self, event):
        # Forward message to the browser WebSocket
        self.send(json.dumps(event))

    # Receive message from WebSocket
    def receive(self, text_data=None, bytes_data=None):
        # If it is text data, that comes from the browser
        if(text_data):
            message = json.loads(text_data)
            # Send message to runner
            async_to_sync(self.channel_layer.group_send)(
                'runner', self.runner_message("actions", message['actions'])
            )
        # Else that comes from the runner
        elif(bytes_data):
            message = json.loads(gzip.decompress(bytes_data))
            group_message = {
                "type": "websocket.message",
                "terminated": message['terminated'],
                "step": message['step'],
                "image": message['image']
            }
            # Send message to room group
            async_to_sync(self.channel_layer.group_send)(
                message['room'], group_message
            )
 




    def disconnect(self, close_code):
        # Leave room group
        async_to_sync(self.channel_layer.group_discard)(
            self.room_name, self.channel_name
        )