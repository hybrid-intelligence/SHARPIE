# chat/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

import cv2
import os
from pettingzoo.mpe import simple_tag_v3

from urllib.parse import parse_qs


class Consumer(AsyncWebsocketConsumer):
    env = {}
    obs = {}
    agent = {}

    static_folder = {}

    is_in_use = {}
    step = {}

    async def connect(self):
        query_params = parse_qs(self.scope["query_string"].decode())
        print(self.scope)
        # Get the room name
        self.room_name = self.scope['session']['room_name']
        self.room_group_name = f"chat_{self.room_name}"
        # Join room group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        if not self.room_name in self.step or self.step[self.room_name]==0:
            # Raise a flag that the room is in use
            self.is_in_use[self.room_name] = True

            # Create the run folder on the server if it does not exist
            app_folder = os.path.dirname(os.path.realpath(__file__))
            self.static_folder[self.room_name] = app_folder+'/static/'+self.scope["path"]+'/'+self.room_name+'/'
            if not os.path.exists(self.static_folder[self.room_name]):
                os.makedirs(self.static_folder[self.room_name])

            # Start the environment and the agent
            self.env[self.room_name] = simple_tag_v3.parallel_env(render_mode="rgb_array", max_cycles=self.scope['session']['max_cycles'])
            # Here we only have humans
            self.agent[self.room_name] = None

            # Get the first observation, render an image and save it on the server
            self.obs[self.room_name], info = self.env[self.room_name].reset()
            cv2.imwrite(self.static_folder[self.room_name]+'step.jpg', self.env[self.room_name].render(), [cv2.IMWRITE_JPEG_QUALITY, 80])

            # Raise a flag that the room is not in use
            self.is_in_use[self.room_name] = False
            self.step[self.room_name] = 0

        await self.accept()

    # Receive message from WebSocket
    async def receive(self, text_data):
        # Don't do anything if a step is already in process
        if(self.is_in_use[self.room_name]):
            return
        # Raise a flag that the room is in use
        self.is_in_use[self.room_name] = True

        # Decode what has been sent by the user
        text_data_json = json.loads(text_data)
        left_action = text_data_json["left"]
        right_action = text_data_json["right"]
        down_action = text_data_json["down"]
        up_action = text_data_json["up"]

        actions = {agent: 0 for agent in self.env[self.room_name].agents}
        played_agent = self.scope['session']['played_agent']
        # Overwrite the action if needed
        if left_action:
            actions[played_agent] = 1
        elif right_action:
            actions[played_agent] = 2
        elif down_action:
            actions[played_agent] = 3
        elif up_action:
            actions[played_agent] = 4
        else:
            actions[played_agent] = 0

        # Perform a step in the environment
        self.obs[self.room_name], reward, terminated, truncated, info = self.env[self.room_name].step(actions)
        # Render an image and save it on the server
        cv2.imwrite(self.static_folder[self.room_name]+'step.jpg', self.env[self.room_name].render(), [cv2.IMWRITE_JPEG_QUALITY, 80])

        # Check if the game is over
        if all(a == 0 for a in terminated):
            self.env[self.room_name].close()

            del self.env[self.room_name]
            del self.obs[self.room_name]
            del self.agent[self.room_name]
            del self.static_folder[self.room_name]
            message = 'done'
        else:
            message = 'not done'

        self.step[self.room_name] += 1
        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name, {"type": "websocket.message", "message": message, "step": self.step[self.room_name]}
        )

        # Check if the game is over
        if all(a == 0 for a in terminated):
            del self.step[self.room_name]

        # Raise a flag that the room is not in use
        self.is_in_use[self.room_name] = False

    # Receive message from room group
    async def websocket_message(self, event):
        message = event["message"]
        step = event["step"]
        # Send message to WebSocket
        await self.send(text_data=json.dumps({"message": message, "step": step}))

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)