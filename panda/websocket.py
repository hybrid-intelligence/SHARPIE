# We import the app_folder because it is needed by the ConsumerTemplate
from .settings import app_folder

from channels.generic.websocket import AsyncWebsocketConsumer
from sharpie.websocket import ConsumerTemplate

import cv2
import os
import gymnasium as gym
from gymnasium import spaces
import json
import numpy as np
import torch
import panda_gym
import gymnasium as gym
import os
import asyncio
import logging

logger = logging.getLogger(__name__)

# environment variable for headless rendering
os.environ["MUJOCO_GL"] = "egl"

# cannot import gymnasium_robotics which is weird :(
# using Panda gym instead. 

class DemonstrationWrapper(gym.Wrapper):
    """Wrapper to add demonstration mode and discretize continuous action spaces"""
    
    def __init__(self, env, discretize_actions=True):
        super().__init__(env)
        self.frozen = False  
        self.human_action = None
        self.discretize_actions = discretize_actions
        self.original_action_space = env.action_space
        
        self.discrete_actions = self._create_discrete_actions()
        
        if self.discretize_actions and isinstance(env.action_space, spaces.Box):
            self.action_space = spaces.Discrete(6)  # 4 directions
            print(f"Original action space shape: {self.original_action_space.shape}")
            print(f"Discretized to: {self.action_space}")
        
        #self.action_mapping = self._setup_action_mapping()
        print("Demo controls: D=toggle demo mode, Arrow keys=movement in demo mode")
        print("Original_action_space: ",self.original_action_space)
        
    def _create_discrete_actions(self):
        """Create 6 discrete actions based on the original action space"""
        if isinstance(self.original_action_space, spaces.Box):
            self.action_dim = self.original_action_space.shape[0]
            print("action_dim: ", self.action_dim)
            action_magnitude = 0.5
            
            if self.action_dim == 2:
                return [
                    np.array([0.0, action_magnitude]),   
                    np.array([0.0, -action_magnitude]),  
                    np.array([-action_magnitude, 0.0]), 
                    np.array([action_magnitude, 0.0])
                ]
            elif self.action_dim >= 3:
                return [
                    np.array([0,  action_magnitude, 0]),  # UP
                    np.array([0, -action_magnitude, 0]),  # DOWN
                    np.array([-action_magnitude, 0, 0]),  # LEFT
                    np.array([ action_magnitude, 0, 0]),  # RIGHT
                    np.array([0, 0,  action_magnitude]),  # Z_UP
                    np.array([0, 0, -action_magnitude])   # Z_DOWN
                ]
            else:
                print(f"Unsupported action dimension: {self.action_dim}")
                return []
        else:
            return list(range(6))  # fallback for non-continuous spaces

    def step(self, action, is_grip, grip_value, frozen ,actor="agent"):

        # Example: Decide gripper state here
        # You can also pass it externally via 'action' if needed.
        # Close gripper
        # grip_value = 0.0  # Open gripper

        if self.discretize_actions and isinstance(self.original_action_space, spaces.Box):

            if isinstance(action, int) and 0 <= action < len(self.discrete_actions):

                continuous_action = self.discrete_actions[action]
            elif isinstance(action, (list, np.ndarray)):
                # Already a continuous action, only take first 3 values
                if is_grip:
                    continuous_action = np.concatenate([action[:3], [action[-1]]])
                else:
                    continuous_action = action[:3]
            else:
                continuous_action = np.zeros(self.original_action_space.shape[0])
            
            # --- Append gripper value ---

            # if the action value is greater 
            if is_grip: # we need to have a grip
                if len(continuous_action) == 3: # human input presumably
                    continuous_action = np.append(continuous_action, grip_value) # TODO gripper needs to be learned as well.
                elif len(continuous_action) > 3: # ai input
                    print("ai action | continuous action: ", continuous_action )
                    #continuous_action[3] = grip_value
                else:
                    raise Exception("continuous action is not normal :)")
                    #continuous_action = np.append(continuous_action, grip_value)
                print(f"Extended action with gripper: {continuous_action}")


            continuous_action = np.clip(continuous_action, 
                                        self.original_action_space.low, 
                                        self.original_action_space.high)
            print(f"Applied action (after clip): {continuous_action}")

            # logger.debug()
            # Call environment step with extended action
            return self.env.step(continuous_action)
        
        else:
            # For non-discretized actions, also append grip value
            continuous_action = np.append(np.array(action), grip_value)
            print(f"Extended action with gripper: {continuous_action}")
            return self.env.step(continuous_action)

    def reset(self, **kwargs):
        self.frozen = False
        return self.env.reset(**kwargs)


# Websocker consumer that inherits from the consumer template
class Consumer(ConsumerTemplate):
    # Here you define the variables you need for your experiment
    app_folder = app_folder
    
    # To separate each rooms, we decided to use dictionaries
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
    frozen = False
    #training metrics
    episode_rewards = {}
    episode_lengths = {}
    total_episodes = {} 
    new_human_input = {}
    button_q = {};
    is_grip = {};
    environment = {};

        
    def prepare_action(self, agent_action, action_space, room_name):
        """
        Ensures the action matches the required shape and constraints of the environment's action space.
        Handles both discrete movement actions and continuous gripper control.

        Args:
            agent_action (np.ndarray | float | int): The action output from the agent.
            action_space (gym.Space): The environment's action space (Box or Discrete).
            room_name (str): The room name to get the correct button_q value.

        Returns:
            np.ndarray | int: Clipped and properly shaped action matching the action space.
        """


        if isinstance(action_space, gym.spaces.Box):
            # For Box action spaces, we need to handle discrete movement + continuous gripper
            
            # Get the button_q value for gripper control
            button_q_value = self.button_q.get(room_name, 1.0)
            
            if isinstance(agent_action, int):
                # Discrete action from human input - convert to continuous movement + gripper
                if 0 <= agent_action < len(self.env[room_name].discrete_actions):
                    # Get the movement part from discrete actions
                    movement_action = self.env[room_name].discrete_actions[agent_action].copy()
                    print(f"Discrete action {agent_action} -> movement: {movement_action}")
                    
                    # Determine required shape
                    required_shape = action_space.shape[0]
                    movement_shape = len(movement_action)
                    
                    if movement_shape < required_shape:
                        # Need to add gripper control
                        pad_width = required_shape - movement_shape
                        padding = np.full(pad_width, button_q_value)
                        final_action = np.concatenate([movement_action, padding])
                        print(f"Added gripper control: {padding}")
                    elif movement_shape > required_shape:
                        final_action = movement_action[:required_shape]
                    else:
                        final_action = movement_action
                        
                else:
                    # Invalid discrete action, use zero movement + gripper
                    final_action = np.zeros(action_space.shape[0])
                    if action_space.shape[0] > 3:  # Assume last dimension is gripper
                        final_action[-1] = button_q_value
                        
            elif isinstance(agent_action, (list, np.ndarray)):
                # Already continuous action from agent
                agent_action = np.asarray(agent_action)
                required_shape = action_space.shape[0]
                current_shape = agent_action.shape[0] if agent_action.ndim > 0 else 1

                if current_shape < required_shape:
                    pad_width = required_shape - current_shape
                    padding = np.full(pad_width, button_q_value)
                    final_action = np.concatenate([agent_action, padding])

                elif current_shape > required_shape:
                    final_action = agent_action[:required_shape]
                else:
                    final_action = agent_action.copy()
                    # Update gripper component if it exists
                    if required_shape > 3:  # Assume last dimension is gripper
                        final_action[-1] = button_q_value
            else:
                # Fallback - create zero action with gripper control
                final_action = np.zeros(action_space.shape[0])
                if action_space.shape[0] > 3:
                    final_action[-1] = button_q_value

            # Clip to action space bounds
            action = np.clip(final_action, action_space.low, action_space.high)
            print("Final Box action:", action)
            return action

        elif isinstance(action_space, gym.spaces.Discrete):
            # Pure discrete action space - no gripper control needed

            if np.isscalar(agent_action):
                scalar_action = int(agent_action)
            elif isinstance(agent_action, np.ndarray):
                if agent_action.size == 1:
                    scalar_action = int(agent_action.item())
                else:
                    # Assume it's a list of scores/Q-values → pick best one
                    scalar_action = int(np.argmax(agent_action))
            else:
                raise TypeError(f"Unsupported type for discrete action: {type(agent_action)}")

            # Clip to ensure within valid range
            action = int(np.clip(scalar_action, 0, action_space.n - 1))
            print("Final Discrete(n) action:", action)
            return action



    # This function is called during the connection with the browser
    async def process_connection(self):
        try:
            print("Process Connection is called.")
            from .agent import DDPGAgent

            self.step[self.room_name] = 0
            self.episode_rewards[self.room_name] = []
            self.episode_lengths[self.room_name] = 0
            self.total_episodes[self.room_name] = 0
            self.button_q[self.room_name] = 1.0;
            self.is_grip[self.room_name] = False;

            # Get session parameters
            max_episode_steps = self.scope['session'].get('max_episode_steps', 50)
            algorithm_name = self.scope['session'].get('algorithm_name', 'DDPG')
            environment = self.scope['session'].get('environment', 'FetchReach-v3')
            self.environment[self.room_name] = environment
            #print("environment: ", environment)
            
            # Start the environment and the agent
            env = gym.make(environment, render_mode = 'rgb_array', max_episode_steps=max_episode_steps)

            self.env[self.room_name] = DemonstrationWrapper(env, discretize_actions= True)
            
            # device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
            logger.info(f"Using device: {self.device}")

            # pick the algorithm
            if algorithm_name == 'DDPG':

                # getting the state dimension
                obs_space = self.env[self.room_name].observation_space

                if isinstance(obs_space, gym.spaces.Dict):
                    state_dim = obs_space['observation'].shape[0] + obs_space['desired_goal'].shape[0]
                else:
                    state_dim = obs_space.shape[0]
                
                action_space = self.env[self.room_name].action_space
                # state_dim = self.env[self.room_name].observation_space.get('observation', None).shape[0] + self.env[self.room_name].observation_space['desired_goal'].shape[0]
                action_dim = 3
                if isinstance(action_space, gym.spaces.Discrete):
                    action_dim = action_space.n  # number of discrete actions
                elif isinstance(action_space, gym.spaces.Box):
                    action_dim = action_space.shape[0]  # number of continuous action dimensions
                else:
                    raise NotImplementedError(f"Unsupported action space: {type(action_space)}")

                original_action_dim = env.action_space.shape[0]

                if original_action_dim > 3:
                    self.is_grip[self.room_name] = True
                    # NOTE if our environment requires a gripper, action dimension should also increase so as to control the robot's gripper.
                    action_dim += 1 
                    print("action_dim: ", action_dim)
                # TODO you can initialize the model with one more action dimension for the gripper. or resort to other methods not currently thought about.
                self.agent[self.room_name] = DDPGAgent(state_dim=state_dim, action_dim=action_dim)
            else:
                raise ValueError(f"not implemented, algorithm: {algorithm_name}")

            # Reset environment and agent
            await self._reset_episode()
            # Load the agent if it does exist

        except Exception as e:
            logger.error(f"Error in process_connection: {e}")
        #self.terminated[self.room_name] = True

    async def _reset_episode(self):

            # Reset environment
            obs, info = self.env[self.room_name].reset()
            self.state[self.room_name] = self._process_observation(obs)
            #print("reset | self.state[self.room_name]: ", self.state[self.room_name])
            # Reset agent noise
            self.agent[self.room_name].noise.reset()
            
            # Reset episode tracking
            self.step[self.room_name] = 0
            self.episode_lengths[self.room_name] = 0
            self.terminated[self.room_name] = False
            self.truncated[self.room_name] = False
            
            logger.info(f"Episode {self.total_episodes[self.room_name]} started")
            
        # except Exception as e:
        #     logger.error(f"Error in _reset_episode: {e}")
        #     self.terminated[self.room_name] = True

    # This function gets the information sent by the browser and processes it
    async def process_inputs(self, text_data):
        try:
            text_data_json = json.loads(text_data)
            left_action = text_data_json.get("left", False)
            right_action = text_data_json.get("right", False)
            down_action = text_data_json.get("down", False)
            up_action = text_data_json.get("up", False)
            z_up_action = text_data_json.get("z_up", False)
            z_down_action = text_data_json.get("z_down", False)

            button_d = text_data_json.get("button_d", False)
            button_q = text_data_json.get('button_q', 1.0)

            # Initialize actions dict if not exists
            if self.room_name not in self.actions:
                self.actions[self.room_name] = {}

            if button_d: # freeze
                if not self.frozen: # if its the first time freezing the environment we have to get out and wait for another human input, (arrow keys)
                    self.frozen = True
                    mode = "DEMO" if self.frozen else "AI"
                    logger.info(f"Mode toggled to: {mode}")
                    # no need to process other stuff
            else: # dont freeze
                if self.frozen:
                    self.frozen = False

            changed = False   
            if self.button_q[self.room_name] == button_q:
                changed = True
            self.button_q[self.room_name] = button_q

        
            print(f"""
                  -----------------------------------------
                    Inside the function process_input:
                    self.frozen = {self.frozen}
                    action = {[left_action, right_action, down_action, up_action, z_up_action, z_down_action, self.button_q[self.room_name]]}
                  ----------------------------------------
                  """)
            # Map browser inputs to actions
            if self.frozen:
                if any([left_action, right_action, down_action, up_action, z_up_action, z_down_action ]):
                    self.new_human_input[self.room_name] = True  # New input received
                    
                    print("action is chosen based on human feedback.", self.frozen)
                    if left_action:
                        self.actions[self.room_name]["human"] = 2
                    elif right_action:
                        self.actions[self.room_name]["human"] = 3
                    elif down_action:
                        self.actions[self.room_name]["human"] = 1
                    elif up_action:
                        self.actions[self.room_name]["human"] = 0
                    elif z_up_action:
                        self.actions[self.room_name]['human'] = 4
                    elif z_down_action:
                        self.actions[self.room_name]['human'] = 5
                    else:
                        self.actions[self.room_name]["human"] = None
                else:
                    self.actions[self.room_name]["human"] = None
                
                # if button q is pressed, register it for later use.
                if changed:
                    self.actions[self.room_name]['grip'] = self.button_q[self.room_name]
                else:
                    self.actions[self.room_name]['grip'] = None

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
        """Perform one step in the environment and train the agent"""
        try:

            room_name = self.room_name

            # Skip if episode is already done
            if self.terminated.get(room_name, False) or self.truncated.get(room_name, False):
                return
            
            # Get current state
            current_state = self.state[self.room_name]
            if current_state is None:
                logger.error("Current state is None")
                return
            # Handle human input vs AI action
            action = None
            # if its human mode and there is no change whatsoever. human did not do anything just skip.
            if self.frozen and self.actions[self.room_name]['grip'] and (not self.actions or not self.actions.get(self.room_name, {})):
                return # no need to do anything this round
            human_action = self.actions.get(self.room_name, {}).get("human", None)

            if self.frozen:
                if self.new_human_input.get(room_name, False) and human_action is not None:
                    action = human_action
                elif self.actions[self.room_name]['grip']: 
                    pass
                else:
                    return
            else:
                action = self.agent[self.room_name].select_action(current_state)
            # Select action using agent

            # Take step in environment
            if self.frozen:
                grip_value = self.button_q[self.room_name]
            else:
                grip_value = 1.0 if action[-1] > 0.0 else -1.0

            try:
                next_obs, reward, terminated, truncated, info = self.env[self.room_name].step(action, self.is_grip[self.room_name], grip_value, self.frozen)

            except Exception as e:
                logger.error(f"[ERROR] Exception during env.step(): {e}")
                import traceback
                traceback.print_exc()
                raise
 
            # Process next observation
            next_state = self._process_observation(next_obs)

            self.next_state[room_name] = next_state
            self.reward[room_name] = reward
            self.terminated[room_name] = terminated
            self.truncated[room_name] = truncated

            # Add experience to replay buffer
            done = terminated or truncated
            self.agent[room_name].replay_buffer.push(
                current_state, action, reward, next_state, done
            )

            # Train agent if enough experiences collected
            if len(self.agent[room_name].replay_buffer) > 1000:
                self.agent[room_name].learn()

            # Update state for next step
            self.state[room_name] = next_state
            self.episode_lengths[room_name] += 1

            # Check if episode is done
            if done:
                await self._handle_episode_end()
            
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

    async def _handle_episode_end(self):
        """Handle end of episode - logging and reset"""
        try:
            room_name = self.room_name
                
                # Calculate episode reward (sum of all rewards in episode)
                # For simplicity, we'll use the final reward, but you might want to track cumulative reward
            episode_reward = self.reward[room_name]
                
            # Log episode results
            self.total_episodes[room_name] += 1
            self.episode_rewards[room_name].append(episode_reward)
                
            # Log every 10 episodes
            if self.total_episodes[room_name] % 1 == 0:
                recent_rewards = self.episode_rewards[room_name][-10:]
                avg_reward = np.mean(recent_rewards)
                logger.info(
                    f"Episode {self.total_episodes[room_name]}: "
                    f"Length: {self.episode_lengths[room_name]}, "
                    f"Reward: {episode_reward:.3f}, "
                    f"Avg (10): {avg_reward:.3f}"
                )
                
            # Reset for next episode
            await self._reset_episode()
                
        except Exception as e:
            logger.error(f"Error in _handle_episode_end: {e}")
        #     self.terminated[self.room_name] = True




    async def process_ouputs(self):
        """Generate output to send back to browser"""
        try:
            room_name = self.room_name
            
            # Render environment and save image
            frame = self.env[self.room_name].render()
            cv2.imwrite(
                self.static_folder[room_name] + 'step.jpg', 
                frame, 
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            
            # Determine message based on episode status
            if self.terminated[room_name] or self.truncated[room_name]:
                message = 'episode_done'
            else:
                message = 'step_done'
            
            # Prepare additional info
            info = {
                'environment': self.environment[self.room_name],
                'step': self.step[room_name],
                'episode': self.total_episodes[room_name],
                'episode_length': self.episode_lengths[room_name],
                'reward': self.reward.get(room_name, 0),
                'buffer_size': len(self.agent[room_name].replay_buffer)
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
                "step": self.step.get(self.room_name, 0),
                'environment': self.environment[self.room_name],
                'episode': self.total_episodes[room_name],

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
                    torch.save({
                        'actor_state_dict': self.agent[room_name].actor.state_dict(),
                        'critic_state_dict': self.agent[room_name].critic.state_dict(),
                        'episode_rewards': self.episode_rewards[room_name],
                        'total_episodes': self.total_episodes[room_name]
                    }, model_path)
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
            self.total_episodes
        ]
        
        for resource_dict in resources_to_clean:
            if room_name in resource_dict:
                del resource_dict[room_name]
        
        logger.info(f"Cleaned up resources for room: {room_name}")

    async def disconnect(self, close_code):
        """Handle client disconnect"""
        try:
            if self.room_name in self.env:
                self._cleanup_room(self.room_name)
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        
        await super().disconnect(close_code)
