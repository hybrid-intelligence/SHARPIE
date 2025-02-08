# chat/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

import os




class ConsumerTemplate(AsyncWebsocketConsumer):
    # Folder where files are stored, this is also use to define the URL of the connection
    app_folder = None
    # Room name used to synchronize "mutiplayer" games
    room_name = None
    room_group_name = None
    # Location where the files are stored on the server
    static_folder = {}
    # Flag used to know if the server is busy with processing inputs
    is_in_use = {}
    # Where the game is over or not
    terminated = {}

    # Here are the functions you will need to write
    async def process_connection(self):
        pass
    async def process_inputs(self, text_data):
        pass
    async def process_step(self):
        pass
    async def process_ouputs(self):
        pass
    async def process_extras(self):
        pass


    async def connect(self):
        # Get the room name, note that app_folder is not defined here because it is defined by each application
        self.room_name = self.scope['session']['room_name']
        self.room_group_name = f"chat_{self.app_folder}_{self.room_name}"
        # Join room group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        # Check if someone did already reset the environment
        if not self.room_name in self.is_in_use:
            # Raise a flag that the room is in use
            self.is_in_use[self.room_name] = True
            self.terminated[self.room_name] = False

            # Create the run folder on the server if it does not exist  
            self.static_folder[self.room_name] = self.app_folder+'/static/'+self.scope["path"]+'/'+self.room_name+'/'
            if not os.path.exists(self.static_folder[self.room_name]):
                os.makedirs(self.static_folder[self.room_name])

            # Initialize the experiment
            await self.process_connection()

            # Raise a flag that the room is not in use
            self.is_in_use[self.room_name] = False

        await self.accept()

    # Receive message from WebSocket
    async def receive(self, text_data):
        # Decode what has been sent by the user
        await self.process_inputs(text_data)

        # Don't do anything if a step is already in process
        if(self.is_in_use[self.room_name]):
            return
        # Raise a flag that the room is in use
        self.is_in_use[self.room_name] = True

        # Perform a step in the environment
        await self.process_step()
        # Send message to room group
        message = await self.process_ouputs()
        await self.channel_layer.group_send(self.room_group_name, message)
        # Perform additional things to do 
        await self.process_extras()

        # Raise a flag that the room is not in use
        self.is_in_use[self.room_name] = False
        # If the game is over, we delete the room to allow to reset it
        if self.terminated[self.room_name]:
            del self.is_in_use[self.room_name]

    # Receive message from room group
    async def websocket_message(self, event):
        # Forward message to the browser WebSocket
        await self.send(text_data=json.dumps(event))

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)