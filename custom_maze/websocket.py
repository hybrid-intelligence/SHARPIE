

# We import the app_folder because it is needed by the ConsumerTemplate

# s a wr (agent human or robot) weights of the algo
# might be action level or epsiode level or session level
# session - demograohics evaluation of the user
# eiposode -> R weights success or failure
from .settings import app_folder
from .agent import DQNAgent
from channels.generic.websocket import AsyncWebsocketConsumer
from sharpie.websocket import ConsumerTemplate

import cv2
import os
import json
import numpy as np
import torch
import panda_gym
import gymnasium as gym
import asyncio
import logging
import pygame
import time
from gym import spaces

import random

def generate_maze(grid_size, complexity=0.75, density=0.75):
    """
    Generates a maze using a modified depth-first search approach.
    complexity: controls number of twists/turns
    density: controls how many walls appear
    """
    # Only odd shapes work for this simple algorithm
    shape = (grid_size, grid_size)
    maze = [[1 for _ in range(shape[0])] for _ in range(shape[1])]

    # Start with empty cells
    for y in range(shape[1]):
        for x in range(shape[0]):
            maze[y][x] = 0

    # Carve passages
    stack = [(0, 0)]
    visited = set()
    visited.add((0, 0))

    def neighbors(cx, cy):
        steps = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(steps)
        for dx, dy in steps:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and (nx, ny) not in visited:
                yield nx, ny, dx, dy

    while stack:
        cx, cy = stack[-1]
        found = False
        for nx, ny, dx, dy in neighbors(cx, cy):
            visited.add((nx, ny))
            maze[cy + dy // 2][cx + dx // 2] = 1  # wall between cells
            stack.append((nx, ny))
            found = True
            break
        if not found:
            stack.pop()

    # Convert carved cells to wall coordinates for env
    walls = set()
    for y in range(shape[1]):
        for x in range(shape[0]):
            if maze[y][x] == 1 and not (x == 0 and y == 0) and not (x == shape[0]-1 and y == shape[1]-1):
                walls.add((x, y))
    return walls

logger = logging.getLogger(__name__)

# environment variable for headless rendering
os.environ["MUJOCO_GL"] = "egl"

class GoalEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, grid_size=10, render_mode="rgb_array"):
        super().__init__()
        
        self.clock = None
        self.grid_size = grid_size
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([grid_size - 1, grid_size - 1])
        self.frozen = False

        # Define maze walls (set of coordinates that are blocked)
        self.walls = generate_maze(self.grid_size)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=grid_size-1, shape=(4,), dtype=np.float32)

        self.window = None
        self.cell_size = 50
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        self.frozen = False
        return self._get_obs(), {}

    def step(self, action, actor="agent"):
        if self.frozen and actor == "agent":
            return self._get_obs(), 0.0, False, False, {}

        # Proposed next position
        new_pos = self.agent_pos.copy()
        if action == 0 and self.agent_pos[1] > 0:  # Up
            new_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:  # Down
            new_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:  # Left
            new_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:  # Right
            new_pos[0] += 1

        # Only update if it's not a wall
        if tuple(new_pos) not in self.walls:
            self.agent_pos = new_pos

        # Check if goal reached
        done = np.array_equal(self.agent_pos, self.goal_pos)

        if done:
            reward = 10.0
        else:
            distance = np.linalg.norm(self.agent_pos - self.goal_pos)
            max_distance = np.linalg.norm([self.grid_size-1, self.grid_size-1])
            reward = -0.1 + (1.0 - distance/max_distance) * 0.05

        return self._get_obs(), reward, done, False, {}
        
    def _get_obs(self):
        return np.concatenate([self.agent_pos, self.goal_pos]).astype(np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            # Create image array for web display
            img = np.ones((self.grid_size * self.cell_size, self.grid_size * self.cell_size, 3), dtype=np.uint8) * 255
            
            # Draw grid
            for x in range(self.grid_size + 1):
                start_pos = (x * self.cell_size, 0)
                end_pos = (x * self.cell_size, self.grid_size * self.cell_size)
                cv2.line(img, start_pos, end_pos, (200, 200, 200), 1)
            
            for y in range(self.grid_size + 1):
                start_pos = (0, y * self.cell_size)
                end_pos = (self.grid_size * self.cell_size, y * self.cell_size)
                cv2.line(img, start_pos, end_pos, (200, 200, 200), 1)
            
            # Inside render() where you draw the grid, after drawing the lines
            for (wx, wy) in self.walls:
                wall_top_left = (wx * self.cell_size, wy * self.cell_size)
                wall_bottom_right = ((wx + 1) * self.cell_size, (wy + 1) * self.cell_size)
                cv2.rectangle(img, wall_top_left, wall_bottom_right, (100, 100, 100), -1)  # gray walls

            # Draw goal (green)
            goal_top_left = (self.goal_pos[0] * self.cell_size, self.goal_pos[1] * self.cell_size)
            goal_bottom_right = ((self.goal_pos[0] + 1) * self.cell_size, (self.goal_pos[1] + 1) * self.cell_size)
            cv2.rectangle(img, goal_top_left, goal_bottom_right, (0, 255, 0), -1)
            
            # Draw agent (red if frozen, blue if AI)
            color = (255, 0, 0) if self.frozen else (0, 0, 255)
            agent_top_left = (self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size)
            agent_bottom_right = ((self.agent_pos[0] + 1) * self.cell_size, (self.agent_pos[1] + 1) * self.cell_size)
            cv2.rectangle(img, agent_top_left, agent_bottom_right, color, -1)
            
            # Add text overlay
            mode_text = "DEMO MODE" if self.frozen else "AI MODE"
            cv2.putText(img, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            return img
        
        elif self.render_mode == "human":
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))

            self.window.fill((255, 255, 255))
            
            if self.clock is None:
                self.clock = pygame.time.Clock()

            # Draw grid
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    rect = pygame.Rect(x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.window, (200, 200, 200), rect, 1)

            # Draw goal
            pygame.draw.rect(
                self.window,
                (0, 255, 0),
                pygame.Rect(self.goal_pos[0]*self.cell_size, self.goal_pos[1]*self.cell_size, 
                           self.cell_size, self.cell_size)
            )

            # draw agent
            color = (255, 0, 0) if self.frozen else (0, 0, 255)
            pygame.draw.rect(
                self.window,
                color,
                pygame.Rect(self.agent_pos[0]*self.cell_size, self.agent_pos[1]*self.cell_size, 
                           self.cell_size, self.cell_size)
            )

            # show mode
            font = pygame.font.Font(None, 36)
            mode_text = "DEMO MODE (Press D to toggle)" if self.frozen else "AI MODE (Press D to toggle)"
            text_surface = font.render(mode_text, True, (0, 0, 0))
            self.window.blit(text_surface, (10, 10))

            pygame.display.flip()
            self.clock.tick(10)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

# WebSocket consumer that inherits from the consumer template
class Consumer(ConsumerTemplate):
    # Here you define the variables you need for your experiment
    app_folder = app_folder
    
    # To separate each room, we decided to use dictionaries
    step = {}
    env = {}
    obs = {}
    state = {}
    next_state = {}
    actions = {}
    agent = {}
    reward = {}
    terminated = {}
    truncated = {}
    epsilon = 0.5
    cum_reward = {}
    gamma = {}
    

    env_type = {}
    # Training metrics
    episode_rewards = {}
    episode_lengths = {}
    total_episodes = {}
    episode_reward = {}
    new_human_input = {}  

    # This function is called during the connection with the browser
    async def process_connection(self):
        try:
            from .agent import DQNAgent
            room_name = self.room_name
            self.step[self.room_name] = 0
            self.episode_rewards[self.room_name] = []
            self.episode_lengths[self.room_name] = 0
            self.total_episodes[self.room_name] = 0
            self.episode_reward[self.room_name] = 0.0
            self.cum_reward[self.room_name] = 0.0
            self.gamma[self.room_name] = 0.9

            # Get session parameters
            max_episode_steps = self.scope['session'].get('max_episode_steps', 100)
            algorithm_name = self.scope['session'].get('algorithm_name', 'DQN')
            self.env_type = self.scope['session'].get('environment', 'pygame')
            
            print("self.env_type: ", self.env_type)
            
            # Start the environment and the agent
            if self.env_type == "pygame":
                self.env[self.room_name] = GoalEnv(render_mode="rgb_array")
                obs, _ = self.env[self.room_name].reset()

            elif self.env_type == 'Frozen Lake':
                print("frozen-1")
                self.env[self.room_name] = gym.make("FrozenLake-v1", render_mode='rgb_array' ,is_slippery=True) # again took me hours and hours :( 
                print("frozen-2")
                obs, _ = self.env[self.room_name].reset()
                print("frozen-3")
            
            elif self.env_type == "PointMaze_UMazeDense-v3":
                self.env[self.room_name] = gym.make('PointMaze_UMazeDense-v3', render_mode='rgb_array')
            
            elif self.env_type == "PointMaze_OpenDense-v3":
                self.env[self.room_name] = gym.make('PointMaze_OpenDense-v3', render_mode='rgb_array')
            
            elif self.env_type == "PointMaze_MediumDense-v3":
                self.env[self.room_name] = gym.make('PointMaze_MediumDense-v3', render_mode='rgb_array')
            
            elif self.env_type == "PointMaze_LargeDense-v3":
                self.env[self.room_name] = gym.make('PointMaze_LargeDense-v3', render_mode='rgb_array')
                
            else: # maybe add other environments
                raise ValueError(f"not implemented, env_type: {self.env_type}")

            if algorithm_name == 'DQN':
                print("self.env[self.room_name].observation_space: ", self.env[self.room_name].observation_space)
                print(self.env[self.room_name].action_space.n)
                if self.env_type == 'pygame':
                    input_dim = self.env[self.room_name].observation_space.shape[0]
                    output_dim = self.env[self.room_name].action_space.n
                elif self.env_type == 'Frozen Lake':
                    input_dim = self.env[self.room_name].observation_space.n
                    output_dim = self.env[self.room_name].action_space.n

                self.agent[self.room_name] = DQNAgent(input_dim, output_dim)
            else:
                raise ValueError(f"not implemented, algorithm: {algorithm_name}")
            
            # device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
            logger.info(f"Using device: {self.device}")

            # Reset environment and agent
            await self._reset_episode()

        except Exception as e:
            logger.error(f"Error in process_connection: {e}")
            self.terminated[self.room_name] = True

    async def _reset_episode(self):
        try:
            # Reset environment
            print("self.env[self.room_name]: ",self.env[self.room_name])
            obs, info = self.env[self.room_name].reset()
            print("obs: ", obs)
            self.state[self.room_name] = self._process_observation(obs)
            
            # Reset episode tracking
            self.step[self.room_name] = 0
            self.episode_lengths[self.room_name] = 0
            self.episode_reward[self.room_name] = 0.0
            self.terminated[self.room_name] = False
            self.truncated[self.room_name] = False
            
            logger.info(f"Episode {self.total_episodes[self.room_name]} started")
            
        except Exception as e:
            logger.error(f"Error in _reset_episode: {e}")
            self.terminated[self.room_name] = True
   

    # This function gets the information sent by the browser and processes it
    async def process_inputs(self, text_data):
        try:
            print("process_inputs")
            text_data_json = json.loads(text_data)
            left_action = text_data_json.get("left", False)
            right_action = text_data_json.get("right", False)
            down_action = text_data_json.get("down", False)
            up_action = text_data_json.get("up", False)
            button_d = text_data_json.get("button_d", False)


            # Initialize actions dict if not exists
            if self.room_name not in self.actions:
                self.actions[self.room_name] = {}

            if button_d: # freeze
                if not self.env[self.room_name].frozen: # if its the first time freezing the environment we have to get out and wait for another human input, (arrow keys)
                    self.env[self.room_name].frozen = True
                    mode = "DEMO" if self.env[self.room_name].frozen else "AI"
                    logger.info(f"Mode toggled to: {mode}")
                    # no need to process other stuff
            
            else: # dont freeze
                self.env[self.room_name].frozen = False


            # Map browser inputs to actions
            if self.env[self.room_name].frozen:
                if any([left_action, right_action, down_action, up_action]):
                    self.new_human_input[self.room_name] = True  # New input received
                    
                    print("action is chosen based on human feedback.", self.env[self.room_name].frozen)
                    if left_action:
                        self.actions[self.room_name]["human"] = 2
                    elif right_action:
                        self.actions[self.room_name]["human"] = 3
                    elif down_action:
                        self.actions[self.room_name]["human"] = 1
                    elif up_action:
                        self.actions[self.room_name]["human"] = 0
                    else:
                        self.actions[self.room_name]["human"] = None
                    print("self.actions[self.room_name]['human']: ", self.actions[self.room_name]["human"])

        except Exception as e:
            logger.error(f"Error in process_inputs: {e}")

    def _process_observation(self, obs):
        """Process observation from environment into state vector"""
        if isinstance(obs, dict):
            # Concatenate observation and desired goal
            state = np.concatenate([
                obs['observation'], 
                obs['desired_goal']
            ], axis=0)
        else:
            state = obs
        return state

    async def process_step(self):
        try:
            print("a1")
            room_name = self.room_name
            print("a2")
            # Skip if episode is already done
            if self.terminated.get(room_name, False) or self.truncated.get(room_name, False):
                return
            
            print("a3")

            # Get current state
            raw_state = self.state.get(self.room_name)
            if raw_state is None:
                logger.error("Current state is None")
                return
            print("a4")
            num_states = None
            if self.env_type == 'pygame':
                num_states = self.env[self.room_name].observation_space.shape[0]
            elif self.env_type == 'Frozen Lake':
                num_states = self.env[self.room_name].observation_space.n

            print("a5")
            current_state = raw_state
            # Handle human input vs AI action
            human_action = self.actions.get(self.room_name, {}).get("human", None)
            print("self.env.get(self.room_name).frozen: ", self.env.get(self.room_name).frozen)
            action = None
            print("state: ",raw_state)
            if self.env_type == 'Frozen Lake':
                # Convert to one-hot for DQN
                num_states = self.env[self.room_name].observation_space.n
                current_state = torch.zeros(1, num_states, dtype=torch.float32)
                current_state[0, raw_state] = 1.0

            if self.env.get(self.room_name).frozen:
                if self.new_human_input.get(room_name, False) and human_action is not None:
                    action = human_action
                    actor = "human"
                    print("human action")
                else: 
                    return
            else:
                action = self.agent[self.room_name].select_action(current_state, use_epsilon=True)
                actor = "agent"
                print("AI action")
            
            print("c1")
            
            print("action:", action)
            # Take step in environment
            if self.env_type == 'pygame':
                next_state, reward, terminated, truncated, _ = self.env[room_name].step(action, actor)
            elif self.env_type == 'Frozen Lake':
                next_state, reward, terminated, truncated, _ = self.env[room_name].step(action)
            done = terminated or truncated
            print("c2")
            # Process next observation
            next_state_processed = self._process_observation(next_state)
            print("next_state_processed:", next_state_processed)
            #next_state_processed = self.one_hot_state(next_state, num_states)

            print("c3")
            # Store transition in agent's memory
            is_demo = (actor == "human")
            self.agent[room_name].store_transition(
                current_state, action, reward, next_state_processed, done, is_demo=is_demo
            )
            print("c4")
            # Train the agent
            loss = 0.0
            if len(self.agent[room_name].memory) >= self.agent[room_name].config['batch_size']:
                print("c11")
                loss = self.agent[room_name].step()
                print("c12")
                if loss is not None:
                    logger.debug(f"Training loss: {loss:.4f}")
            print("c5")
            # Update episode tracking
            self.episode_reward[room_name] += reward
            self.episode_lengths[room_name] += 1
       
            # Store transition data for output
            self.next_state[room_name] = next_state_processed
            self.reward[room_name] = reward
            self.terminated[room_name] = terminated
            self.truncated[room_name] = truncated
            
            # Update state for next step
            self.state[room_name] = next_state_processed

            # Check if episode is done
            if done:
                await self._handle_episode_end()
           
            # Decay epsilon
            self.agent[room_name].decay_epsilon()

            # Reset human action
            if room_name in self.actions and "human" in self.actions[room_name]:
                self.actions[room_name]["human"] = None
                print("helloooo this is the step function :) ")
            
            # calculating the cumulative reward so far.
            self.cum_reward[self.room_name] =  self.reward[self.room_name] + self.cum_reward[self.room_name] * self.gamma[self.room_name]

            agent_config = self.agent[room_name].get_debug_param()
            logger.info(f"""
                episode: {self.total_episodes[self.room_name]},
                step number: {self.step[self.room_name]}
                state (position, velocity): {self.state[room_name]}
                action_taken: {action}
                policy: 
                immediate reward: {self.reward[room_name]}
                cumulative_reward: {self.cum_reward[self.room_name]}
                state value: 
                action value: 
                loss: {loss}
                gradient norm:  {agent_config['grad_norm']}
                learning rate: {agent_config['learning_rate']}
                user input: 
                user input type: 
            """)


        except Exception as e:
            logger.error(f"Error in process_step: {e}")
            self.terminated[self.room_name] = True

    async def _handle_episode_end(self):
        """Handle end of episode - logging and reset"""
        try:
            room_name = self.room_name
            
            # Log episode results
            self.total_episodes[room_name] += 1
            self.episode_rewards[room_name].append(self.episode_reward[room_name])
            
            # Log every episode
            if self.total_episodes[room_name] % 1 == 0:
                recent_rewards = self.episode_rewards[room_name][-10:]
                avg_reward = np.mean(recent_rewards)
                logger.info(
                    f"Episode {self.total_episodes[room_name]}: "
                    f"Length: {self.episode_lengths[room_name]}, "
                    f"Reward: {self.episode_reward[room_name]:.3f}, "
                    f"Avg (10): {avg_reward:.3f}, "
                    f"Epsilon: {self.agent[room_name].epsilon:.3f}"
                )
            
            # Reset for next episode
            await self._reset_episode()
            
        except Exception as e:
            logger.error(f"Error in _handle_episode_end: {e}")
            self.terminated[self.room_name] = True

    async def process_ouputs(self):
        """Generate output to send back to browser"""
        try:
            room_name = self.room_name
            
            # Render environment and save image
            print("outputs --1")
            frame = self.env[room_name].render()
            print("outputs --2")

            if frame is not None:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                print("outputs --3")
                cv2.imwrite(
                    self.static_folder[room_name] + 'step.jpg', 
                    frame_bgr, 
                    [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                print("outputs --4")
            
            # Determine message based on episode status
            if self.terminated[room_name] or self.truncated[room_name]:
                message = 'episode_done'
            else:
                message = 'step_done'
            
            # Prepare additional info
            info = {
                'step': self.step[room_name],
                'episode': self.total_episodes[room_name],
                'episode_length': self.episode_lengths[room_name],
                'reward': self.reward.get(room_name, 0),
                'episode_reward': self.episode_reward.get(room_name, 0),
                'epsilon': self.agent[room_name].epsilon,
                'buffer_size': len(self.agent[room_name].memory),
                'demo_buffer_size': len(self.agent[room_name].demo_memory),
                'frozen': self.env[room_name].frozen
            }
            
            # Add recent performance if available
            if len(self.episode_rewards[room_name]) > 0:
                recent_rewards = self.episode_rewards[room_name][-10:]
                info['avg_reward_10'] = np.mean(recent_rewards)
            
            return {
                "type": "websocket.message", 
                "message": message,
                **info
            }
            
        except Exception as e:
            logger.error(f"Error in process_outputs: {e}")
            return {
                "type": "websocket.message", 
                "message": "error",
                "error": str(e),
                "step": self.step.get(self.room_name, 0)
            }

    async def process_extras(self):
        """Clean up resources and increment step counter"""
        try:
            room_name = self.room_name
            
            # If episode is done, we've already reset in _handle_episode_end
            # If not done, increment step counter
            if not (self.terminated.get(room_name, False) or self.truncated.get(room_name, False)):
                self.step[room_name] += 1
            
            # Clean up if connection is terminated
            if self.terminated.get(room_name, False) and room_name in self.env:
                # Save model before cleanup (optional)
                try:
                    model_path = self.static_folder[room_name] + 'trained_model.pth'
                    self.agent[room_name].save_model(model_path)
                    logger.info(f"Model saved to {model_path}")
                except Exception as save_error:
                    logger.error(f"Failed to save model: {save_error}")
                
                # Clean up resources
                self._cleanup_room(room_name)
                
        except Exception as e:
            logger.error(f"Error in process_extras: {e}")

    def _cleanup_room(self, room_name):
        """Clean up all resources for a specific room"""
        resources_to_clean = [
            self.step, self.env, self.obs, self.state, self.next_state,
            self.actions, self.reward, self.terminated, self.truncated,
            self.agent, self.episode_rewards, self.episode_lengths,
            self.total_episodes, self.episode_reward
        ]
        
        for resource_dict in resources_to_clean:
            if room_name in resource_dict:
                del resource_dict[room_name]
        
        # Close environment
        if room_name in self.env:
            self.env[room_name].close()
        
        logger.info(f"Cleaned up resources for room: {room_name}")

    async def disconnect(self, close_code):
        """Handle client disconnect"""
        try:
            if self.room_name in self.env:
                self._cleanup_room(self.room_name)
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        
        await super().disconnect(close_code)