# chat/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

import os

from amaze.simu.controllers.tabular import TabularController
from amaze import Maze, Robot, Simulation, InputType, OutputType, StartLocation, MazeWidget, qt_application
import random

ALPHA, GAMMA = 0.1, 0.5
ROBOT = Robot.BuildData(inputs=InputType.DISCRETE, outputs=OutputType.DISCRETE)


class Consumer(AsyncWebsocketConsumer):
    env = {}
    obs = {}
    action = {}
    agent = {}

    
    obs_ = {}
    action_ = {}

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

            rng = random.Random(0)
            # Start the environment and the agent
            maze_data = Maze.BuildData(
                width=self.scope["session"]['width'],
                height=self.scope["session"]['height'],
                seed=self.scope["session"]['seed'],
                unicursive=self.scope["session"]['unicursive'],
                start=rng.choice([sl for sl in StartLocation]),
                p_lure=self.scope["session"]['lures'],
                p_trap=self.scope["session"]['traps'],
            )
            maze = Maze.generate(maze_data)
            self.env[self.room_name] = Simulation(maze, ROBOT, save_trajectory=True)

            # Load the agent if it does exist
            if os.path.exists(self.static_folder[self.room_name]+'agent/'):
                self.agent[self.room_name] = TabularController(robot_data=ROBOT, epsilon=0.1*random.random(), seed=self.scope["session"]['seed']).load(self.static_folder[self.room_name]+'agent')
            else:
                self.agent[self.room_name] = TabularController(robot_data=ROBOT, epsilon=0.1*random.random(), seed=self.scope["session"]['seed'])

            # Get the first observation, render an image and save it on the server
            self.obs[self.room_name] = self.env[self.room_name].generate_inputs().copy()

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
        if(text_data_json["reward"]) != '':
            reward = float(text_data_json["reward"])
        if self.scope["session"]['train'] and self.step[self.room_name] > 1 and text_data_json["reward"] != '':
            # Learn from the reward given by the user
            self.agent[self.room_name].q_learning(self.obs[self.room_name], 
                                                  self.action[self.room_name], 
                                                  reward, 
                                                  self.obs_[self.room_name], 
                                                  self.action_[self.room_name], 
                                                  alpha=ALPHA, gamma=GAMMA)
        
        if self.step[self.room_name] > 0:
            self.action_[self.room_name] = self.action[self.room_name]
            self.obs_[self.room_name] = self.obs[self.room_name]

        # Overwrite the action if needed
        if self.scope["session"]['train']==False:
            action = self.agent[self.room_name].greedy_action(self.obs[self.room_name])
        else:
            action = self.agent[self.room_name](self.obs[self.room_name])

        # Perform a step in the environment
        reward = self.env[self.room_name].step(action)
        self.action[self.room_name] = action
        self.obs[self.room_name] = self.env[self.room_name].observations.copy()

        # Render an image and save it on the server
        rng = random.Random(0)
        maze_data = Maze.BuildData(
            width=self.scope["session"]['width'],
            height=self.scope["session"]['height'],
            seed=self.scope["session"]['seed'],
            unicursive=self.scope["session"]['unicursive'],
            start=rng.choice([sl for sl in StartLocation]),
            p_lure=self.scope["session"]['lures'],
            p_trap=self.scope["session"]['traps'],
        )
        maze = Maze.generate(maze_data)
        # Call PyQt needed for rendering
        app = qt_application()
        MazeWidget.static_render_to_file(maze=maze, path=self.static_folder[self.room_name]+'maze.png', size=1000, robot=False, solution=True, dark=True)
        MazeWidget.plot_trajectory(
            simulation=self.env[self.room_name],
            size=500,
            path=self.static_folder[self.room_name]+'step.png',
        )

        # Check if the game is over
        if self.env[self.room_name].done():
            self.agent[self.room_name].save(self.static_folder[self.room_name]+'agent')

            del self.obs[self.room_name]
            del self.agent[self.room_name]
            del self.static_folder[self.room_name]
            message = 'done'
        else:
            message = 'not done'

        self.step[self.room_name] += 1
        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name, {"type": "websocket.message", "message": message, "step": self.step[self.room_name], "reward": reward}
        )

        # Check if the game is over
        if self.env[self.room_name].done():
            del self.env[self.room_name]
            del self.step[self.room_name]

        # Raise a flag that the room is not in use
        self.is_in_use[self.room_name] = False

    # Receive message from room group
    async def websocket_message(self, event):
        message = event["message"]
        step = event["step"]
        reward = event["reward"]
        # Send message to WebSocket
        await self.send(text_data=json.dumps({"message": message, "step": step, "reward": reward}))

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)