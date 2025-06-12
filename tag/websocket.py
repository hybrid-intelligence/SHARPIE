# We import the app_folder because it is needed by the ConsumerTemplate
from .settings import app_folder

from sharpie.websocket import ConsumerTemplate

import cv2
import os
import json
from pettingzoo.mpe import simple_tag_v3

# Websocker consumer that inherits from the consumer template
class Consumer(ConsumerTemplate):
    # Here you define the variables you need for your experiment
    app_folder = app_folder
    # To separate each rooms, we decided to use dictionaries
    step = {}

    env = {}
    obs = {}
    agent = {}
    actions = {}

    # This function is called during the connection with the browser
    async def process_connection(self):
        # Initialize the number of steps
        self.step[self.room_name] = 0
        
        # Start the environment and the agent
        self.env[self.room_name] = simple_tag_v3.parallel_env(render_mode="rgb_array", max_cycles=self.scope['session']['max_cycles'])
        # Here we only have humans
        self.agent[self.room_name] = None

        # Get the first observation, render an image and save it on the server
        self.obs[self.room_name], info = self.env[self.room_name].reset()        
        self.actions[self.room_name] = {agent: 0 for agent in self.env[self.room_name].agents}

    # This function gets the information sent by the browser and processes it
    async def process_inputs(self, text_data):
        text_data_json = json.loads(text_data)
        left_action = text_data_json["left"]
        right_action = text_data_json["right"]
        down_action = text_data_json["down"]
        up_action = text_data_json["up"]

        played_agent = self.scope['session']['played_agent']
        # Overwrite the action if needed
        if left_action:
            self.actions[self.room_name][played_agent] = 1
        elif right_action:
            self.actions[self.room_name][played_agent] = 2
        elif down_action:
            self.actions[self.room_name][played_agent] = 3
        elif up_action:
            self.actions[self.room_name][played_agent] = 4
        else:
            self.actions[self.room_name][played_agent] = 0

    # This function performs a step in the experiment
    async def process_step(self):
        # Perform a step in the environment
        self.obs[self.room_name], reward, terminated, truncated, info = self.env[self.room_name].step(self.actions[self.room_name])
        self.terminated[self.room_name] = all(a == 0 for a in terminated)

    # This function generates the rendered image and returns the information sent back to the browser
    async def process_ouputs(self):
        # Get the rendered image
        img = self.env[self.room_name].render()
        
        # Get the user-controlled agent's position from current observation
        played_agent = self.scope['session']['played_agent']
        # The position is in the observation at index 2 (after velocity and self position)
        agent_pos = self.obs[self.room_name][played_agent][2:4] 
        
        height, width = img.shape[:2]
        
        # Convert position to pixel coordinates using actual image dimensions
        x = int((agent_pos[0] + 1) * width / 2)  # Scale from [-1,1] to [0,width]
        y = int((agent_pos[1] + 1) * height / 2)  # Scale from [-1,1] to [0,height]
        
        # Add position indicator
        font_scale = min(width, height) / 1000 
        font_thickness = max(1, int(font_scale * 2)) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Format position text (2 decimal places)
        pos_text = f"({agent_pos[0]:.2f}, {agent_pos[1]:.2f})"
        
        (text_width, text_height), _ = cv2.getTextSize(pos_text, font, font_scale, font_thickness)
        text_x = x - text_width // 2
        text_y = y - int(height * 0.05) 
        
        cv2.putText(img, pos_text, (text_x, text_y), font, font_scale, (255, 255, 0), font_thickness)
        cv2.imwrite(self.static_folder[self.room_name]+'step.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # Check if the game is over
        if self.terminated[self.room_name]:
            message = 'done'
        else:
            message = 'not done'
        # Send message to room group
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
