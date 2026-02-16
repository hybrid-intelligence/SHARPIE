#!/usr/bin/env python

"""Client using the asyncio API."""

import numpy as np
import base64
import cv2
import time
import json
from multiprocessing import Process
import argparse
import logging
import importlib
import sys
import os

from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosedError



def sanitize_data(data):
    if isinstance(data, np.int64):
        return data.item()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(v) for v in data]
    else:
        return data

def send_message(websocket, env, step_count, terminated, truncated, obs, actions, reward):
    frame = env.render()
    step_count += 1

    # Encode numpy frame as base64, makes it more compact for transfer
    _, buffer = cv2.imencode('.jpeg', frame.astype(np.uint8))
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Create message to send to server
    group_message = {
        "type": "broadcast",
        "terminated": terminated,
        "truncated": truncated,
        "step": step_count,
        "observations": sanitize_data(obs),
        "rewards": sanitize_data(reward),
        "actions": sanitize_data(actions),
        "image": image_base64
    }
    # Send message to server
    websocket.send(json.dumps(group_message))
    return step_count

def receive_message(websocket, agents_settings):
    actions = {}
    for agent_name, agent_config in agents_settings.items():
        # If the participant is supposed to give some input
        if agent_config['participant']:
            message = json.loads(websocket.recv())
            if 'error' in message.keys():
                logging.info(f"Message from room: {message['error']}")
                exit(1)

            actions[message['role']] = message['action']
    return actions






def train_policies(policy_modules, policy_checkpoint_intervals, prev_obs, actions, obs, reward, terminated, truncated, agents_settings):
    """
    Train policies based on their checkpoint_interval settings.

    Args:
        policy_modules: Dictionary mapping agent names to their policy modules
        policy_checkpoint_intervals: Dictionary mapping agent names to checkpoint intervals
        prev_obs: Previous observation(s) before the step
        actions: Actions taken
        obs: Current observation(s) after the step
        reward: Reward(s) received
        terminated: Whether the episode has ended
        truncated: Whether the episode was truncated
        agents_settings: Dictionary of agent settings
    """
    for agent_name, policy_module in policy_modules.items():
        checkpoint_interval = policy_checkpoint_intervals[agent_name]

        # Determine if training should occur based on checkpoint_interval
        # Note: step_count is managed by the caller for interval checking
        # This function is called when training should occur
        if checkpoint_interval == 0:
            # Never train
            continue

        # Get the reward for this agent (handle both single and multi-agent rewards)
        if isinstance(reward, dict):
            agent_reward = reward.get(agent_name, 0)
        else:
            agent_reward = reward

        # Get the observation for this agent
        if isinstance(obs, dict):
            agent_next_obs = obs.get(agent_name, obs)
        else:
            agent_next_obs = obs

        # Get previous observation for this agent
        if isinstance(prev_obs, dict):
            agent_prev_obs = prev_obs.get(agent_name, prev_obs)
        else:
            agent_prev_obs = prev_obs

        # Get action for this agent
        agent_action = actions.get(agent_name, 0)

        # Try to call update() method (from policy template)
        if hasattr(policy_module, 'update'):
            try:
                policy_module.update(agent_prev_obs, agent_action, agent_reward, terminated or truncated, agent_next_obs)
            except Exception as e:
                logging.warning(f"Policy update failed for {agent_name}: {e}")


def get_policy_actions(obs, policy_modules, participant_inputs=None, agents_settings=None):
    """
    Get actions from policies for all agents that have policies.

    Args:
        obs: Current observation(s) from the environment
        policy_modules: Dictionary mapping agent names to their policy modules
        participant_inputs: Dictionary mapping agent names to participant input values
        agents_settings: Dictionary mapping agent names to their settings (including inputs_type)

    Returns:
        Dictionary mapping agent names to actions
    """
    actions = {}
    for agent_name, policy_module in policy_modules.items():
        # Get the observation for this agent (assuming multi-agent observations)
        if isinstance(obs, dict):
            agent_obs = obs.get(agent_name, obs)
        else:
            agent_obs = obs

        # Get inputs_type for this agent (default to 'actions' in the agent data model)
        inputs_type = agents_settings[agent_name]['inputs_type']

        # Handle different inputs_type cases
        if inputs_type == 'other' and participant_inputs and agent_name in participant_inputs:
            # Pass input as extra parameter to predict
            actions[agent_name] = policy_module.predict(agent_obs, participant_input=participant_inputs[agent_name])
        else:
            # Standard predict (no extra parameters)
            actions[agent_name] = policy_module.predict(agent_obs)
    return actions







def load_episode(websocket):
    # Ask for settings
    websocket.send(json.dumps({'type': 'private', 'message': 'settings'}))
    # Wait for settings
    environment_settings = json.loads(websocket.recv())
    # Save all files to disk
    for value in environment_settings['files'].values():
        if value['content']:
            with open(value['path'], 'w') as f:
                f.write(value['content'])
    agents_settings = json.loads(websocket.recv())
    # Save all files to disk
    for agent in agents_settings.values():
        if 'policy' in agent:
            for value in agent['policy']['files'].values():
                if value['content']:
                    with open(value['path'], 'w') as f:
                        f.write(value['content'])
    experiment_settings = json.loads(websocket.recv())
    return environment_settings, agents_settings, experiment_settings



def run_episode(websocket, environment_settings, agents_settings, experiment_settings):
    # Load the module from the file path
    spec = importlib.util.spec_from_file_location("environment", environment_settings['files']['environment']['path'])
    env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_module)
    # Load the module from the file path
    policy_modules = {}
    policy_checkpoint_intervals = {}
    for agent_name, agent_config in agents_settings.items():
          if 'policy' in agent_config:
              policy_file_path = agent_config['policy']['files']['policy']['path']
              # Add the directory of the policy file to sys.path so related scripts can be imported
              policy_dir = os.path.dirname(os.path.abspath(policy_file_path))
              if policy_dir not in sys.path:
                  sys.path.insert(0, policy_dir)
              spec = importlib.util.spec_from_file_location(f"policy", policy_file_path)
              policy_module = importlib.util.module_from_spec(spec)
              spec.loader.exec_module(policy_module)
              policy_modules[agent_name] = policy_module.policy
              # Store checkpoint interval for training
              policy_checkpoint_intervals[agent_name] = agent_config['policy'].get('checkpoint_interval', 0)

    # Initialize environment
    env = env_module.environment
    obs, info = env.reset()

    # Initialize variables
    step_count = 0
    terminated = False
    truncated = False
    reward = 0
    # Get initial actions for agents
    actions = {}
    # Store previous observation for training
    prev_obs = obs

    while not terminated and not truncated:
        start_time = time.time()

        # Send data to server
        step_count = send_message(websocket, env, step_count, terminated, truncated, obs, actions, reward)
        # Wait for inputs from participants
        participant_inputs = receive_message(websocket, agents_settings)
        # Override reward for agents with inputs_type == 'reward'
        for agent_name, participant_input in participant_inputs.items():
            if agents_settings[agent_name].get('inputs_type') == 'reward':
                if isinstance(reward, dict):
                    reward[agent_name] = participant_input
                else:
                    reward = participant_input

        # Get actions from policies (handles 'actions' and 'other' inputs_type)
        actions = get_policy_actions(obs, policy_modules, participant_inputs, agents_settings)
        # Override action for agents with inputs_type == 'actions'
        for agent_name, participant_input in participant_inputs.items():
            if agents_settings[agent_name].get('inputs_type') == 'actions':
                # Get the default input value from the agent's keyboard_inputs mapping
                default_input = agents_settings[agent_name]['keyboard_inputs'].get('default', 0)
                # Do not override policy action if default is provided from participant
                if agent_name in actions and participant_input == default_input:
                    continue
                if isinstance(actions, dict):
                    actions[agent_name] = participant_input
                else:
                    actions = participant_input
        # Perform a step in the environment
        obs, reward, terminated, truncated, info = env.step(actions)

        # Train policies based on checkpoint_interval
        for agent_name, checkpoint_interval in policy_checkpoint_intervals.items():
            if checkpoint_interval == 0:
                # Never train
                continue
            elif checkpoint_interval == -2:
                # Train at end of episode only
                if terminated or truncated:
                    train_policies(policy_modules, policy_checkpoint_intervals, prev_obs, actions, obs, reward, terminated, truncated, agents_settings)
                    break
            elif checkpoint_interval > 0 and step_count % checkpoint_interval == 0:
                # Train every N steps
                train_policies(policy_modules, policy_checkpoint_intervals, prev_obs, actions, obs, reward, terminated, truncated, agents_settings)
                break

        # Store current observation as previous for next step
        prev_obs = obs

        loop_time = time.time() - start_time
        target_frame_time = 1.0 / (experiment_settings['target_fps'] * 1.1)    # Add a small buffer to account for sleep overhead
        # If wait_for_inputs is enabled add a forced pause
        if experiment_settings['wait_for_inputs']:
            forced_pause = 1.0 / experiment_settings['target_fps']
            time.sleep(forced_pause)
        # Sleep to maintain target FPS (only if wait_for_inputs is not enabled)
        else:
            time.sleep(max(0, target_frame_time - loop_time)) 

    # Send final message to server indicating termination
    send_message(websocket, env, step_count, terminated, truncated, obs, actions, reward)


def start_experiment(hostname, port, connection_key, link, room):
    try:
        with connect(f"ws://{hostname}:{port}/experiment/{link}/run/{room}", additional_headers={"authorization": f"{connection_key}"}) as websocket:
            logging.info(f"Connected to experiment {link}")
            environment_settings, agents_settings, experiment_settings = load_episode(websocket)
            logging.info(f"Starting experiment for room {room}")
            run_episode(websocket, environment_settings, agents_settings, experiment_settings)
        logging.info(f"Connection to {link} closed")
    except ConnectionRefusedError:
        logging.warning(f"Connection to {link} refused")







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment runner.")
    parser.add_argument("--hostname", type=str, default="localhost", help="Hostname of the server")
    parser.add_argument("--port", type=int, default=8000, help="Port of the server")
    parser.add_argument("--connection-key", type=str, default='', help="Connection key for the server")
    parser.add_argument("command", type=str, help="Command to run: 'runserver' or 'test'")

    args = parser.parse_args()
    hostname = args.hostname
    port = args.port
    connection_key = args.connection_key

    sleeptime = 1.0
    while True:
        try:
            with connect(f"ws://{hostname}:{port}/runner/connection", additional_headers={"authorization": f"{connection_key}"}) as websocket:
                sleeptime = 1.0
                logging.info(f"Connected to server")
                while True:
                    websocket.send(json.dumps({"status": "idle"}))
                    message = json.loads(websocket.recv())
                    if 'experiment' in message.keys():
                        logging.info(f"Starting experiment {message['experiment']}")
                        p = Process(target=start_experiment, args=(hostname, port, connection_key, message['experiment'], message['room']))
                        p.start()
                        p.join()
                        if p.exitcode:
                            logging.warning(f"Episode failed with return code {p.exitcode}")
                        else:
                            logging.info(f"Episode succeeded")
                    else:
                        logging.info(f"No pending session")
                    time.sleep(1)
        except ConnectionClosedError:
            logging.warning(f"Connection to server closed")
            continue
        except ConnectionRefusedError:
            logging.warning(f"Connection to server refused")
            time.sleep(sleeptime)
            sleeptime = min(60.0, sleeptime * 2)
            continue
