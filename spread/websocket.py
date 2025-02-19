# We import the app_folder because it is needed by the ConsumerTemplate
from .settings import app_folder

from channels.generic.websocket import AsyncWebsocketConsumer
from sharpie.websocket import ConsumerTemplate

import cv2
import os
import json
from pettingzoo.mpe import simple_spread_v3

import tensorflow as tf
from .agent import Agent, agent_info, MIN_EPSILON, EPSILON_DECAY

# Websocker consumer that inherits from the consumer template
class Consumer(ConsumerTemplate):
    # Here you define the variables you need for your experiment
    app_folder = app_folder
    # To separate each rooms, we decided to use dictionaries
    step = {}

    env = {}
    obs = {}
    actions = {}
    agent = {}

    # This function is called during the connection with the browser
    async def process_connection(self):
        # Initialize the number of steps
        self.step[self.room_name] = 0

        # Start the environment and the agent
        self.env[self.room_name] = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=self.scope['session']['max_cycles'])
        # Load the agent if it does exist
        if os.path.exists(self.static_folder[self.room_name]+'agent.keras'):
            self.agent[self.room_name] = Agent(agent_info, self.env[self.room_name])
            self.agent[self.room_name].model = tf.keras.models.load_model(self.static_folder[self.room_name]+'agent.keras')
        else:
            self.agent[self.room_name] = Agent(agent_info, self.env[self.room_name])

        # Start the agent
        actions = self.agent[self.room_name].agent_start()
        # Perform a step as it is needed for this agent
        _, _ = self.env[self.room_name].reset()
        next_states, reward, terminated, _, _ = self.env[self.room_name].step(actions)
        self.obs[self.room_name] = (next_states, reward, terminated)

    # This function gets the information sent by the browser and processes it
    async def process_inputs(self, text_data):
        (next_states, rewards, terminated) = self.obs[self.room_name]
        self.actions[self.room_name] = self.agent[self.room_name].agent_step(next_states, rewards, terminated)
        if self.agent[self.room_name].epsilon > MIN_EPSILON:
            self.agent[self.room_name].epsilon *= EPSILON_DECAY
            self.agent[self.room_name].epsilon = max(MIN_EPSILON, self.agent[self.room_name].epsilon)
        
        text_data_json = json.loads(text_data)
        left_action = text_data_json["left"]
        right_action = text_data_json["right"]
        down_action = text_data_json["down"]
        up_action = text_data_json["up"]

        # Overwrite the action if needed
        if left_action:
            self.actions[self.room_name]['agent_0'] = 1
        elif right_action:
            self.actions[self.room_name]['agent_0'] = 2
        elif down_action:
            self.actions[self.room_name]['agent_0'] = 3
        elif up_action:
            self.actions[self.room_name]['agent_0'] = 4
        else:
            self.actions[self.room_name]['agent_0'] = 0

    # This function performs a step in the experiment
    async def process_step(self):
        next_states, reward, terminated, _, _ = self.env[self.room_name].step(self.actions[self.room_name])
        self.obs[self.room_name] = (next_states, reward, terminated)
        self.terminated[self.room_name] = all(a == 0 for a in terminated)

    # This function generates the rendered image and returns the information sent back to the browser
    async def process_ouputs(self):
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
            del self.step[self.room_name]

            del self.env[self.room_name]
            del self.obs[self.room_name]
            del self.agent[self.room_name]
            del self.actions[self.room_name]
        else:
            self.step[self.room_name] += 1
