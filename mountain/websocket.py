# We import the app_folder because it is needed by the ConsumerTemplate
from .settings import app_folder

from sharpie.websocket import ConsumerTemplate

import cv2
import os
import ezpickle
import gymnasium as gym
import numpy as np
import json

from .agent import DISCRETE_OBSERVATION_SPACE_SIZE, epsilon, get_discrete_state, update_q_table

# Websocker consumer that inherits from the consumer template
class Consumer(ConsumerTemplate):
    # Here you define the variables you need for your experiment
    app_folder = app_folder
    # To separate each rooms, we decided to use dictionaries
    step = {}
    env = {}
    agent = {}
    action = {}
    obs = {}

    # This function is called during the connection with the browser
    async def process_connection(self):
        # Initialize the number of steps
        self.step[self.room_name] = 0

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

    # This function gets the information sent by the browser and processes it
    async def process_inputs(self, text_data):
        # Decode what has been sent by the user
        text_data_json = json.loads(text_data)
        left_action = text_data_json["left"]
        right_action = text_data_json["right"]

        # Overwrite the action if needed
        if left_action:
            self.action[self.room_name] = 0
        elif right_action:
            self.action[self.room_name] = 2
        elif np.random.random() > epsilon or self.scope["session"]['train']==False:
            self.action[self.room_name] = np.argmax(self.agent[self.room_name][self.obs[self.room_name]])
        else:
            self.action[self.room_name] = np.random.randint(0, self.env[self.room_name].action_space.n)

    # This function performs a step in the experiment
    async def process_step(self):
        # Perform a step in the environment
        observation, reward, terminated, truncated, info = self.env[self.room_name].step(self.action[self.room_name])
        # Update the Q-table of the agent
        self.obs[self.room_name] = get_discrete_state(observation)
        update_q_table(self.obs[self.room_name], self.action[self.room_name], reward, self.agent[self.room_name], terminated, truncated, observation)
        self.terminated[self.room_name] = (terminated or truncated or observation[0] > 0.5)

    # This function generates the rendered image and returns the information sent back to the browser
    async def process_ouputs(self):
        # Render an image and save it on the server
        cv2.imwrite(self.static_folder[self.room_name]+'step.jpg', self.env[self.room_name].render(), [cv2.IMWRITE_JPEG_QUALITY, 80])

        # Check if the game is over
        if self.terminated[self.room_name]:
            message = 'done'
        else:
            message = 'not done'
        # Send message to room group
        # The returned value should be a dictionnary with the 
        #   type, 
        #   message, 
        #   step, 
        #   and anything else you would like to send
        return {"type": "websocket.message", 
                "message": message, 
                "step": self.step[self.room_name]}
    
    # This function takes care of anything else we need to do at the end of the request
    async def process_extras(self):
        if self.terminated[self.room_name]:
            # Delete the variables from memory
            ezpickle.pickle_data(self.agent[self.room_name], self.static_folder[self.room_name]+'agent.pkl', overwrite=True)
            self.env[self.room_name].close()

            del self.step[self.room_name]

            del self.env[self.room_name]
            del self.obs[self.room_name]
            del self.agent[self.room_name]
            del self.action[self.room_name]
        else:
            self.step[self.room_name] += 1
