# We import the app_folder because it is needed by the ConsumerTemplate
from .settings import app_folder
from .models import Info

from sharpie.websocket import ConsumerTemplate
from django.contrib.auth.models import User
from channels.db import database_sync_to_async

import os
import json

from amaze.simu.controllers.tabular import TabularController
from amaze import Maze, Robot, Simulation, InputType, OutputType, StartLocation, MazeWidget, qt_application
import random

ALPHA, GAMMA = 0.1, 0.5
ROBOT = Robot.BuildData(inputs=InputType.DISCRETE, outputs=OutputType.DISCRETE)

# Websocker consumer that inherits from the consumer template
class Consumer(ConsumerTemplate):
    # Here you define the variables you need for your experiment
    app_folder = app_folder
    # To separate each rooms, we decided to use dictionaries
    step = {}

    env = {}
    maze = {}
    obs = {}
    action = {}
    agent = {}
    
    reward = {}
    obs_ = {}
    action_ = {}

    changed = {}
    user_feedback = {}  # New variable to store user feedback

    @database_sync_to_async
    def update_info(self):
        new_info = Info(user=self.scope["user"], room=self.room_name, reward=self.reward[self.room_name], changed=self.changed[self.room_name])
        new_info.save()

    # This function is called during the connection with the browser
    async def process_connection(self):
        # Initialize the number of steps
        self.step[self.room_name] = 0
        self.user_feedback[self.room_name] = None  # Initialize user feedback

        rng = random.Random(0)
        # Start the environment and the agent
        maze_data = Maze.BuildData(
            width=self.scope["session"]['width'],
            height=self.scope["session"]['height'],
            seed=self.scope["session"]['seed'],
            unicursive=self.scope["session"]['unicursive'].lower() == 'true',  # Convert string to boolean
            start=rng.choice([sl for sl in StartLocation]),
            p_lure=self.scope["session"]['lures'],
            p_trap=self.scope["session"]['traps'],
        )
        self.maze[self.room_name] = Maze.generate(maze_data)
        self.env[self.room_name] = Simulation(self.maze[self.room_name], ROBOT, save_trajectory=True)

        # Load the agent if it does exist
        if os.path.exists(self.static_folder[self.room_name]+'agent/'):
            self.agent[self.room_name] = TabularController(robot_data=ROBOT, epsilon=0.1*random.random(), seed=self.scope["session"]['seed']).load(self.static_folder[self.room_name]+'agent')
        else:
            self.agent[self.room_name] = TabularController(robot_data=ROBOT, epsilon=0.1*random.random(), seed=self.scope["session"]['seed'])

        # Get the first observation, render an image and save it on the server
        self.obs[self.room_name] = self.env[self.room_name].generate_inputs().copy()

    # This function gets the information sent by the browser and processes it
    async def process_inputs(self, text_data):
        self.changed[self.room_name] = False
        # Decode what has been sent by the user
        text_data_json = json.loads(text_data)
        
        # Handle user feedback
        if "feedback" in text_data_json:
            self.user_feedback[self.room_name] = float(text_data_json["feedback"])
            # Use the feedback as the reward
            self.reward[self.room_name] = self.user_feedback[self.room_name]
            # Don't trigger step automatically - wait for user to click Next
            self.changed[self.room_name] = True
        elif text_data_json["reward"] != '':
            self.changed[self.room_name] = True
            self.reward[self.room_name] = float(text_data_json["reward"])

    # This function performs a step in the experiment
    async def process_step(self):
        if self.scope["session"]['train'] and self.step[self.room_name] > 1:
            # Learn from the reward given by the user
            self.agent[self.room_name].q_learning(self.obs[self.room_name], 
                                                  self.action[self.room_name], 
                                                  self.reward[self.room_name], 
                                                  self.obs_[self.room_name], 
                                                  self.action_[self.room_name], 
                                                  alpha=ALPHA, gamma=GAMMA)
        
        if self.step[self.room_name] > 0:
            self.action_[self.room_name] = self.action[self.room_name]
            self.obs_[self.room_name] = self.obs[self.room_name]

        # Overwrite the action if needed
        if self.scope["session"]['train']==False:
            self.action[self.room_name] = self.agent[self.room_name].greedy_action(self.obs[self.room_name])
        else:
            self.action[self.room_name] = self.agent[self.room_name](self.obs[self.room_name])

        # Perform a step in the environment
        env_reward = self.env[self.room_name].step(self.action[self.room_name])
        # Only use environment reward if no user feedback is provided
        if self.user_feedback[self.room_name] is None:
            self.reward[self.room_name] = env_reward
        self.obs[self.room_name] = self.env[self.room_name].observations.copy()
        self.terminated[self.room_name] = self.env[self.room_name].done()
        # Reset user feedback after using it
        self.user_feedback[self.room_name] = None

    # This function generates the rendered image and returns the information sent back to the browser
    async def process_ouputs(self):
        # Call PyQt needed for rendering
        app = qt_application()
        MazeWidget.static_render_to_file(maze=self.maze[self.room_name], path=self.static_folder[self.room_name]+'maze.jpg', size=1000, robot=False, solution=True, dark=True)
        MazeWidget.plot_trajectory(
            simulation=self.env[self.room_name],
            size=500,
            path=self.static_folder[self.room_name]+'step.jpg',
        )
        # Store the data into the DB
        await self.update_info()

        # Check if the game is over
        if self.terminated[self.room_name]:
            message = 'done'
        else:
            message = 'not done'
        # Send message to room group
        return {"type": "websocket.message", 
                "message": message, 
                "step": self.step[self.room_name],
                "reward": self.reward[self.room_name]}
    
    # This function takes care of anything else we need to do at the end of the request
    async def process_extras(self):
        if self.terminated[self.room_name]:
            # Delete the variables from memory
            self.agent[self.room_name].save(self.static_folder[self.room_name]+'agent')
            del self.step[self.room_name]
            del self.maze[self.room_name]
            del self.env[self.room_name]
            del self.agent[self.room_name]
            del self.obs[self.room_name]
            del self.obs_[self.room_name]
            del self.action[self.room_name]
            del self.action_[self.room_name]
            del self.reward[self.room_name]
            del self.user_feedback[self.room_name]
        else:
            self.step[self.room_name] += 1
