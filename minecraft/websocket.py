# We import the app_folder because it is needed by the ConsumerTemplate
from .settings import app_folder

from sharpie.websocket import ConsumerTemplate

import argparse
import json
import os

import cv2
import numpy as np
import torch
from steve1.data.text_alignment.vae import load_vae_model
from steve1.utils.mineclip_agent_env_utils import load_mineclip_agent_env

from steve1.config import PRIOR_INFO, DEVICE
from steve1.utils.embed_utils import get_prior_embed


in_model = app_folder+"/data/weights/vpt/2x.model"
in_weights = app_folder+"/data/weights/steve1/steve1.weights" 
prior_weights = app_folder+"/data/weights/steve1/steve1_prior.pt" 

prior = load_vae_model(PRIOR_INFO)
cond_scale = 6.0
seed = None


# Websocker consumer that inherits from the consumer template
class Consumer(ConsumerTemplate):
    # Here you define the variables you need for your experiment
    app_folder = app_folder
    # To separate each rooms, we decided to use dictionaries
    step = {}

    env = {}
    mineclip = {}
    agent = {}
    obs = {}
    prompt = {}
    prompt_embed = {}

    # This function is called during the connection with the browser
    async def process_connection(self):
        # Initialize the number of steps
        self.step[self.room_name] = 0

        # Start the environment
        self.agent[self.room_name], self.mineclip[self.room_name], self.env[self.room_name] = load_mineclip_agent_env(in_model, in_weights, seed, cond_scale)
        
        self.obs[self.room_name] = self.env[self.room_name].reset()
        if seed is not None:
            self.env[self.room_name].seed(seed)

        # Set initial prompt
        self.prompt[self.room_name] = self.scope["session"]['initial_prompt']
        self.prompt_embed[self.room_name] = get_prior_embed(self.prompt[self.room_name], self.mineclip[self.room_name], prior, DEVICE)

        # Get the first observation, render an image and save it on the server
        cv2.imwrite(self.static_folder[self.room_name]+'step.jpg', self.obs[self.room_name]['pov'], [cv2.IMWRITE_JPEG_QUALITY, 80])
        # Initialize the number of steps and terminated variable
        self.step[self.room_name] = 0

    # This function gets the information sent by the browser and processes it
    async def process_inputs(self, text_data):
        # Decode what has been sent by the user
        text_data_json = json.loads(text_data)
        prompt = text_data_json["prompt"].strip().lower()

        # Reset the agent or env if prompted
        if prompt == 'reset agent':
            self.agent[self.room_name].reset(cond_scale)
        elif prompt == 'reset env':
            self.obs[self.room_name] = self.env[self.room_name].reset()
            if seed is not None:
                self.env[self.room_name].seed(seed)
        elif prompt == 'stop game':
            self.terminated[self.room_name] = True
        else:
            if self.prompt[self.room_name] != prompt:
                # Use prior to get the prompt embed
                self.prompt_embed[self.room_name] = get_prior_embed(prompt, self.mineclip[self.room_name], prior, DEVICE)
                self.prompt[self.room_name] = prompt


    # This function performs a step in the experiment
    async def process_step(self):
        if not self.terminated[self.room_name]:
            with torch.cuda.amp.autocast():
                minerl_action = self.agent[self.room_name].get_action(self.obs[self.room_name], self.prompt_embed[self.room_name])
                self.obs[self.room_name], _, self.terminated[self.room_name], _ = self.env[self.room_name].step(minerl_action)

    # This function generates the rendered image and returns the information sent back to the browser
    async def process_ouputs(self):
        # Render an image and save it on the server
        cv2.imwrite(self.static_folder[self.room_name]+'step.jpg', self.obs[self.room_name]['pov'], [cv2.IMWRITE_JPEG_QUALITY, 80])
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
            del self.agent[self.room_name]
            del self.obs[self.room_name]
            del self.prompt[self.room_name]
            del self.mineclip[self.room_name]
        else:
            self.step[self.room_name] += 1
