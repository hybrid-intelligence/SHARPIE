# chat/consumers.py
import json
from mysite.websocket import ConsumerTemplate

import cv2
import os
import ezpickle
import gym as gym2
import minerl
import numpy as np



class Consumer(ConsumerTemplate):
    env = {}
    obs = {}
    
    step = {}

    async def process_connection(self):
        # Start the environment
        self.env[self.room_name] = gym2.make("MineRLBasaltFindCave-v0")
        # Get the first observation, render an image and save it on the server
        self.obs[self.room_name] = self.env[self.room_name].reset()
        cv2.imwrite(self.static_folder[self.room_name]+'step.jpg', self.obs[self.room_name]['pov'], [cv2.IMWRITE_JPEG_QUALITY, 80])
        # Initialize the number of steps and terminated variable
        self.step[self.room_name] = 0

    async def process_inputs(self, text_data):
        # No inputs in this experiment
        return

    async def process_step(self):
        # Generate a random action
        action = self.env[self.room_name].action_space.sample()
        # Prevent from escaping the game
        action["ESC"] = 0
        self.obs[self.room_name], reward, self.terminated[self.room_name], infos =  self.env[self.room_name].step(action)

    async def process_ouputs(self):
        # Render an image and save it on the server
        save_folder = self.static_folder[self.room_name]+'step.jpg'
        cv2.imwrite(save_folder, self.obs[self.room_name]['pov'], [cv2.IMWRITE_JPEG_QUALITY, 80])
        # Check if the game is over
        if self.terminated[self.room_name]:
            message = 'done'
        else:
            message = 'not done'
        # Send message to room group
        return {"type": "websocket.message", 
                "message": message, 
                "step": self.step[self.room_name]}
    
    async def process_extras(self):
        if self.terminated[self.room_name]:
            self.env[self.room_name].close()
            # Delete the environment from memory
            del self.env[self.room_name]
            del self.obs[self.room_name]
            del self.step[self.room_name]
        else:
            self.step[self.room_name] += 1
