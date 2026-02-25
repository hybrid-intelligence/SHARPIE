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
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        # Covers all numpy scalar types (np.int64, np.float32, np.bool_, etc.)
        return data.item()
    elif isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
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
            if 'action' in message.keys():
                actions[message['role']] = message['action']
    return actions

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

def load_environment(env_config):
    spec = importlib.util.spec_from_file_location("environment", env_config['files']['environment']['path'])
    env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_module)
    return env_module.environment

def load_policies(agents_settings):
    policy_modules = {}
    policy_intervals = {}
    for agent_name, agent_config in agents_settings.items():
        if 'policy' in agent_config:
            path = agent_config['policy']['files']['policy']['path']
            policy_dir = os.path.dirname(os.path.abspath(path))
            if policy_dir not in sys.path:
                sys.path.insert(0, policy_dir)
            
            spec = importlib.util.spec_from_file_location(f"policy_{agent_name}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            policy_modules[agent_name] = module.policy
            policy_intervals[agent_name] = agent_config['policy'].get('checkpoint_interval', 0)
    return policy_modules, policy_intervals

def override_actions(agents_settings, participant_inputs, policy_actions):
    """Applies human overrides to policy-suggested actions."""
    final_actions = policy_actions.copy() if isinstance(policy_actions, dict) else policy_actions
    
    for agent_name, p_input in participant_inputs.items():
        if agents_settings[agent_name].get('inputs_type') == 'actions':
            default_val = agents_settings[agent_name].get('keyboard_inputs', {}).get('default', 0)
            # Only override if the participant actually provided a non-default input
            if agent_name in final_actions and p_input != default_val:
                continue
            if isinstance(final_actions, dict):
                final_actions[agent_name] = p_input
            else:
                final_actions = p_input
    return final_actions

def override_rewards(agents_settings, participant_inputs, env_reward):
    """Replaces environment rewards with human feedback where configured."""
    final_reward = env_reward.copy() if isinstance(env_reward, dict) else env_reward
    
    for agent_name, p_input in participant_inputs.items():
        if agents_settings[agent_name].get('inputs_type') == 'reward':
            if isinstance(final_reward, dict):
                final_reward[agent_name] = p_input
            else:
                final_reward = p_input
    return final_reward

def get_agent_data(agent_name, data, default=0):
    """Safely extracts agent-specific data from potentially nested structures."""
    if isinstance(data, dict):
        return data.get(agent_name, default)
    return data

def update_single_agent(agent_name, policy_module, s, a, r, s_prime, done):
    """Executes the update call for a single policy module."""
    if not hasattr(policy_module, 'update'):
        return

    # Extract agent-specific slices of the transition
    agent_s = get_agent_data(agent_name, s)
    agent_a = get_agent_data(agent_name, a)
    agent_r = get_agent_data(agent_name, r)
    agent_s_prime = get_agent_data(agent_name, s_prime)

    try:
        policy_module.update(agent_s, agent_a, agent_r, done, agent_s_prime)
    except Exception as e:
        logging.warning(f"Policy update failed for {agent_name}: {e}")

def train_policies(policy_modules, intervals, step, s, a, r, term, trunc, settings):
    """
    Coordinates training for all agents. 
    Determines eligibility based on intervals and delegates the update.
    """
    done = term or trunc
    
    for agent_name, policy_module in policy_modules.items():
        interval = intervals.get(agent_name, 0)

        # 1. Eligibility Check
        should_train = False
        if interval == -2 and done:
            should_train = True
        elif interval > 0 and step % interval == 0:
            should_train = True
        
        # 2. Execution
        if should_train:
            update_single_agent(agent_name, policy_module, s, a, r, s_prime=r, done=done)

def run_episode(websocket, environment_settings, agents_settings, experiment_settings):
    # 1. Initialization
    env = load_environment(environment_settings)
    policy_modules, checkpoint_intervals = load_policies(agents_settings)
    
    current_obs, info = env.reset()
    step_count = 0
    terminated = truncated = False
    
    # Buffers to track the transition waiting for human feedback
    last_obs = current_obs
    last_actions = {}
    last_env_reward = 0

    # 2. Episode Loop
    while not (terminated or truncated):
        start_time = time.time()

        # Update UI with current state (result of the previous step)
        step_count = send_message(websocket, env, step_count, terminated, truncated, 
                                 current_obs, last_actions, last_env_reward)
        
        # Capture Human Input (Feedback for the transition that just happened)
        participant_inputs = receive_message(websocket, agents_settings)
        
        # --- PHASE A: Resolve Reward and Train ---
        # The human's reward input evaluates the S_t -> A_t -> S_t+1 transition
        final_reward = override_rewards(agents_settings, participant_inputs, last_env_reward)

        if step_count > 0:
            train_policies(
                policy_modules, checkpoint_intervals, step_count, last_obs, last_actions, final_reward, terminated, truncated, agents_settings)

        # --- PHASE B: Act and Step ---
        # Prepare for the next transition
        last_obs = current_obs
        
        # Determine Actions (Policy first, then Human Override)
        raw_policy_actions = get_policy_actions(current_obs, policy_modules, participant_inputs, agents_settings)
        last_actions = override_actions(agents_settings, participant_inputs, raw_policy_actions)
        
        # Advance Environment
        current_obs, last_env_reward, terminated, truncated, info = env.step(last_actions)

        # --- PHASE C: Timing ---
        target_fps = experiment_settings['target_fps']
        if experiment_settings['wait_for_inputs']:
            time.sleep(1.0 / target_fps)
        else:
            loop_time = time.time() - start_time
            time.sleep(max(0, (1.0 / (target_fps * 1.1)) - loop_time))

    # 3. Finalization
    # Send the final state to server/UI and check for final end-of-episode training
    final_reward = override_rewards(agents_settings, {}, last_env_reward) 
    send_message(websocket, env, step_count, terminated, truncated, current_obs, last_actions, final_reward)
    
    train_policies(
        policy_modules, checkpoint_intervals, step_count, last_obs, last_actions, final_reward, terminated, truncated, agents_settings
    )


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
