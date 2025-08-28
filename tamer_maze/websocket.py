# We import the app_folder because it is needed by the ConsumerTemplate
from .settings import app_folder
from .agent import DQNAgent 
from channels.generic.websocket import AsyncWebsocketConsumer
from sharpie.websocket import ConsumerTemplate
from channels.db import database_sync_to_async
from .models import Info_Tamer

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
from scipy.stats import expon  # for random variables like exponential
from scipy.integrate import quad # for the integration purposes

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

        # actions: 0 up, 1 down, 2 left, 3 right
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
        action = int(action)
        print("ENV | Step function | action: ",action)
        if self.frozen and actor == "agent":
            return self._get_obs(), 0.0, False, False, {}

        # Move agent
        if action == 0 and self.agent_pos[1] > 0:  # Up
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:  # Down
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:  # Left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:  # Right
            self.agent_pos[0] += 1

        # Check if goal reached
        done = np.array_equal(self.agent_pos, self.goal_pos)
        
        # Consistent reward structure
        if done:
            reward = 10.0  # Goal reached
        else:
            # Distance-based reward to guide learning
            distance = np.linalg.norm(self.agent_pos - self.goal_pos)
            max_distance = np.linalg.norm([self.grid_size-1, self.grid_size-1])
            reward = -0.1 + (1.0 - distance/max_distance) * 0.05  # Small positive shaping

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
    env_type = "pygame"
    epsilon = 0.5
    E_history = {}
    H_history = {}
    Events = {}
    
    # Training metrics
    episode_rewards = {}
    episode_lengths = {}
    total_episodes = {}
    episode_reward = {}
    new_human_input = {}  
    changed = {}
    index = {}
    policy_type = {}

    user_feedback = {}  # New variable to store user feedback


    @database_sync_to_async
    def update_info(self):
        new_info = Info_Tamer(user=self.scope["user"], room=self.room_name, reward=self.reward[self.room_name], changed=self.changed[self.room_name])
        new_info.save()

    # This function is called during the connection with the browser
    async def process_connection(self):
        try:
            print("process connection")
            from .agent import DQNAgent
            room_name = self.room_name
            self.index[self.room_name] = -1
            self.step[self.room_name] = 0
            self.E_history[self.room_name] = [] # [(start_time, end_time, state, action, reward)]
            self.H_history[self.room_name] = [] # [(time, value)]
            self.episode_rewards[self.room_name] = []
            self.episode_lengths[self.room_name] = 0
            self.total_episodes[self.room_name] = 0
            self.episode_reward[self.room_name] = 0.0
            self.changed[self.room_name] = False
            self.Events[self.room_name] = [] # events are loaded here.
            self.policy_type[self.room_name] = 'e-greedy'


            print("connection | cp--1")

#             E_hist = []   # [(start_time, end_time, state, action, reward)]
#             H_hist = []   # [(time, value)]

            # Get session parameters
            max_episode_steps = self.scope['session'].get('max_episode_steps', 50)
            algorithm_name = self.scope['session'].get('algorithm_name', 'DQN')
            
            print("connection | cp--2")
            
            # Start the environment and the agent
            if self.env_type == "pygame":
                self.env[self.room_name] = GoalEnv(render_mode="rgb_array")
                obs, _ = self.env[self.room_name].reset()
                
            else: # maybe add other environments
                raise ValueError(f"not implemented, env_type: {self.env_type}")
            
            print("connection | cp--3")

            if algorithm_name == 'DQN':
                input_dim = self.env[self.room_name].observation_space.shape[0]
                output_dim = self.env[self.room_name].action_space.n
                
                self.agent[self.room_name] = DQNAgent(input_dim, output_dim)
            else:
                raise ValueError(f"not implemented, algorithm: {algorithm_name}")
            
            print("connection | cp--4")
            
            # device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
            logger.info(f"Using device: {self.device}")
            
            print("connection | cp--5")
            
            # Reset environment and agent
            await self._reset_episode()

            print("connection | cp--6")

        except Exception as e:
            logger.error(f"Error in process_connection: {e}")
            self.terminated[self.room_name] = True
# nice expression -> bugs me a bit
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
            current_time = time.time()
            print("process_inputs")

            self.changed[self.room_name] = False
            text_data_json = json.loads(text_data)

            if "feedback" in text_data_json:
                self.user_feedback[self.room_name] = float(text_data_json["feedback"])
                # Use the feedback as the reward
                self.reward[self.room_name] = self.user_feedback[self.room_name]
                # Don't trigger step automatically - wait for user to click Next
                self.changed[self.room_name] = True
               
                # Append the human feedback history with it.
                self.H_history[self.room_name].append((current_time, self.reward[self.room_name]))
                print("self.H_history[self.room_name]: ", self.H_history[self.room_name])

                    # ADD THIS: Train on most recent action immediately
                if self.E_history[self.room_name]:
                    recent_event = self.E_history[self.room_name][-1]
                    _, _, state, action, _ = recent_event
                    self.agent[self.room_name].train_rewardmodel(
                        torch.Tensor(state), action, self.reward[self.room_name]
                    )
            # elif text_data_json["reward"] != '':
            #     self.changed[self.room_name] = True
            #     self.reward[self.room_name] = float(text_data_json["reward"])
        
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
    
    # specifies the human reward delay distribution functions
    def f_delay(self, x):
        return expon.pdf(x, loc=0.2, scale=0.7) # scale means dist. drops off too quickly. loc means the minimum delay
    
    def p_targets(self, h_time, e_start, e_end):
        prob,_ =  quad(self.f_delay, h_time - e_start, h_time - e_end )
        return prob
      # threshold
      

    async def process_step(self):  # [(start_time, end_time, state, action, reward)]
        try:
            room_name = self.room_name
            # self.step[self.room_name] += 1
            # Skip if episode is already done
            if self.terminated.get(room_name, False) or self.truncated.get(room_name, False):
                return
            i = self.index[self.room_name] + 1
            # i <- i + 1
            self.index[self.room_name] = i # updating the original index here as well, as we use it as default at the beginning of the each step call.
            
            current_time = time.time()
            print("---cp-1")
            if i - 1 >= 0:
                # e i - 1 t_t = t_curr
                event = self.Events[self.room_name][i-2]
                self.Events[self.room_name][i-2] = (event[0], current_time, event[2], event[3], event[4])

            print("---cp-2")
            print("i: ", i)
            print("len(self.E_history[self.room_name]): ",len(self.E_history[self.room_name]))
            print("self.E_history[self.room_name]: ", self.E_history[self.room_name])
            print("len(self.Events[self.room_name): ", len(self.Events[self.room_name]))
            if i > 0:
                event = self.Events[self.room_name][i-1] # i counts the element number in the list. we have to refer to ith element  with i-1
                self.Events[self.room_name][i-1] = (current_time, event[1], event[2], event[3], event[4])
                print("---cp-3")
                epsilon_p = 0.05  
                
                # main update of the algorithm
                for e in self.E_history[self.room_name][:]:  # [(start_time, end_time, state, action, reward)]
                    print("---cp-4")
                    print(e)
                    print(type(e))
                    e_start, e_end, state, action, reward = e
                    print("---cp-5")
                    if all(self.p_targets(h_time, e_start, e_end) < epsilon_p for h_time, _ in self.H_history[self.room_name] if h_time > current_time): # if the threshold is not exceeded for all human rewards, that means we are far away from the intended event.
                        print("---cp-6")
                        total_reward = sum(
                            h_val * self.p_targets(h_time, e_start, e_end)
                            for h_time, h_val in self.H_history[self.room_name]
                        )
                        print(f"---cp-7 --- total reward ${total_reward}" )
                        self.agent[self.room_name].train_rewardmodel(torch.Tensor(e[2]), e[3], total_reward) # state, action, human_reward
                        print("---cp-8")
                        self.E_history[self.room_name].remove(e)
                        print("---cp-9")
                        

                # #---------------------------------------------
                # # t -> t + 1
                # if self.step[self.room_name] >= 2:
                #     if self.changed[self.room_name]: # if reward is given
                #         pre_rew = self.reward[self.room_name] # get the reward
                #         if pre_rew != 0: # if reward is meaningful
                #             #  update the parameters of hte reward model.
                #             self.agent[self.room_name].train_rewardmodel(torch.Tensor(current_state), action ,pre_rew)
                # #---------------------------------------------
    

                for h in self.H_history[self.room_name][:]:
                # remove fully credited rewards from H_hist
                    h_time, h_val = h
                    print(h)
                    print("type(e_start): ",type(e_start))
                    print("type(e_end): ",type(e_end))
                    print("self.E_history: ",self.E_history)
                    if all(abs(quad(self.f_delay, h_time - e_end, h_time - e_start)[0] - 1) < 1e-6 for e_start, e_end,*_ in self.E_history[self.room_name]):
                        self.H_history[self.room_name].remove(h)


            # Get current state
            current_state = self.state.get(self.room_name, None)
            if current_state is None:
                logger.error("Current state is None")
                return
            
            # choose action
            action = self.agent[self.room_name].select_action(torch.Tensor(current_state), use_epsilon=True)

            # E_hist <- E_hist U {e_i} NOTE end time is not important here, we ll update at the beginning of the next 'step' function
            print("len(self.E_history[self.room_name]): ",len(self.E_history[self.room_name]))
            new_event = (current_time, current_time ,current_state, action, 0 )
            self.E_history[self.room_name].append(new_event)  # [(start_time, end_time, state, action, reward)]
            self.Events[self.room_name].append(new_event) # adding to the event lsit

            # take action
            next_state, _, terminated, truncated, _ = self.env[room_name].step(action)
            done = terminated or truncated

            # Process next observation
            next_state_processed = self._process_observation(next_state)

            # # Store transition in agent's memory
            # self.agent[room_name].store_transition(
            #     current_state, action, reward, next_state_processed, done
            # )
            print("self.agent[room_name].memory: ", type(self.agent[room_name].memory))
            # Train the agent
            loss = 0.0
            if len(self.agent[room_name].memory) >= self.agent[room_name].config['batch_size']:
                loss = self.agent[room_name].step()
                if loss is not None:
                    logger.debug(f"Training loss: {loss:.4f}")
           
            # Update episode tracking # we can add some more stuff here.
            # self.episode_reward[room_name] += reward
            self.episode_lengths[room_name] += 1
       
            # Store transition data for output
            self.next_state[room_name] = next_state_processed
            
            # self.reward[room_name] = reward
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

            # logging processes begin here
            agent_config = self.agent[room_name].get_debug_param()
            logger.info(f"""
                episode: {self.total_episodes[self.room_name]},
                step number: {self.step[self.room_name]}
                state (position, velocity): {self.state[room_name]}
                action_taken: {action}
                policy: 
                immediate reward: 
                cumulative_reward:
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
           # self.episode_rewards[room_name].append(self.episode_reward[room_name])
            
            # # Log every episode
            # if self.total_episodes[room_name] % 1 == 0:
            #     recent_rewards = self.episode_rewards[room_name][-10:]
            #     avg_reward = np.mean(recent_rewards)
            #     logger.info(
            #         f"Episode {self.total_episodes[room_name]}: "
            #         f"Length: {self.episode_lengths[room_name]}, "
            #         f"Reward: {self.episode_reward[room_name]:.3f}, "
            #         f"Avg (10): {avg_reward:.3f}, "
            #         f"Epsilon: {self.agent[room_name].epsilon:.3f}"
            #     )
            
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
            frame = self.env[room_name].render()
            if frame is not None:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    self.static_folder[room_name] + 'step.jpg', 
                    frame_bgr, 
                    [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
            # Store the data into the DB
            await self.update_info()
            
            # Determine message based on episode status
            if self.terminated[room_name] or self.truncated[room_name]:
                message = 'episode_done'
            else:
                message = 'step_done'
            
            # Prepare additional info
            info = {
                'step': self.step[self.room_name],
                'episode': self.total_episodes[self.room_name],
                'episode_length': self.episode_lengths[self.room_name],
                'reward': self.reward.get(self.room_name, 0),
                'episode_reward': self.episode_reward.get(self.room_name, 0),
                'epsilon': self.agent[self.room_name].epsilon,
                'buffer_size': len(self.agent[self.room_name].memory),
                'frozen': self.env[self.room_name].frozen,

            }
            
            # Add recent performance if available
            if len(self.episode_rewards[room_name]) > 0:
                recent_rewards = self.episode_rewards[self.room_name][-10:]
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
