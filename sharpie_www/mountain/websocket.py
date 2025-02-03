# chat/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

import cv2
import os
import ezpickle
import gymnasium as gym
import numpy as np

from .agent import *


class Consumer(AsyncWebsocketConsumer):
    env = {}
    obs = {}
    agent = {}

    static_folder = {}

    is_in_use = {}
    step = {}

    async def connect(self):
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
            self.env[self.room_name] = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=self.scope["session"]['goal_velocity'])
            # Load the agent if it does exist
            if os.path.exists(self.static_folder[self.room_name]+'agent.pkl'):
                self.agent[self.room_name] = ezpickle.unpickle_data(self.static_folder[self.room_name]+'agent.pkl')
            else:
                self.agent[self.room_name] = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBSERVATION_SPACE_SIZE + [self.env[self.room_name].action_space.n]))

            # Get the first observation, render an image and save it on the server
            observation, info = self.env[self.room_name].reset()
            self.obs[self.room_name] = get_discrete_state(observation)
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

        # Overwrite the action if needed
        if left_action:
            action = 0
        elif right_action:
            action = 2
        elif np.random.random() > epsilon or self.scope["session"]['train']==False:
            action = np.argmax(self.agent[self.room_name][self.obs[self.room_name]])
        else:
            action = np.random.randint(0, self.env[self.room_name].action_space.n)

        # Perform a step in the environment
        observation, reward, terminated, truncated, info = self.env[self.room_name].step(action)
        # Update the Q-table of the agent
        self.obs[self.room_name] = get_discrete_state(observation)
        update_q_table(self.obs[self.room_name], action, reward, self.agent[self.room_name], terminated, truncated, observation)
        # Render an image and save it on the server
        cv2.imwrite(self.static_folder[self.room_name]+'step.jpg', self.env[self.room_name].render(), [cv2.IMWRITE_JPEG_QUALITY, 80])

        # Check if the game is over
        if terminated or truncated or observation[0] > 0.5:
            ezpickle.pickle_data(self.agent[self.room_name], self.static_folder[self.room_name]+'agent.pkl', overwrite=True)
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
        if terminated or truncated or observation[0] > 0.5:
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