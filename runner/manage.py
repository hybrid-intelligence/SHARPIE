#!/usr/bin/env python

"""Client using the asyncio API."""

import numpy as np
import base64
import cv2
import time
import json
import gzip
import os
import sys
from multiprocessing import Process
import argparse
import logging

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
    elif isinstance(data, tuple):
        return tuple(sanitize_data(v) for v in data)
    else:
        return data
    
def action_mapping(actions):
    if len(actions) == 1:
        return actions['agent_0']
    return actions

def send_message(websocket, room, env, step_count, termination_condition, terminated, truncated, obs, actions, reward):
    frame = env.render()
    step_count += 1

    # Encode numpy frame as base64, makes it more compact for transfer
    _, buffer = cv2.imencode('.jpeg', frame.astype(np.uint8))
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Create message to send to server
    group_message = {
        "room": room,
        "terminated": termination_condition(terminated, truncated),
        "step": step_count,
        "observations": sanitize_data(obs),
        "rewards": sanitize_data(reward),
        "actions": sanitize_data(actions),
        "image": image_base64
    }
    compressed_message = gzip.compress(json.dumps(group_message).encode('utf-8'))
    # Send message to server
    websocket.send(compressed_message)
    return step_count

def receive_message(websocket, room, users_needed, inputs):
    for i in range(users_needed):
        message = json.loads(websocket.recv())
        if 'message' in message.keys():
            logging.info(f"Message from room {room}: {message['message']}")
            if message['message'] == 'A user has disconnected':
                exit(1)
        inputs[message['session']['agent']] = message['actions']

def generate_actions(ai_agents, obs, evaluate, actions):
    print(f"[kgd-debug] {ai_agents=} {actions=}")
    for ai_agent in ai_agents:
        if evaluate:
            actions[ai_agent.name] = ai_agent.predict(obs)
        else:
            actions[ai_agent.name] = ai_agent.sample(obs)

def train_agents(ai_agents, state, actions, reward, done, next_state):
    for ai_agent in ai_agents:
        ai_agent.train(state, action_mapping(actions), reward, done, next_state)

def run_episode(websocket, room, users_needed, type, target_fps, train, evaluate):
    from environment import environment, termination_condition, input_mapping

    try:
        from agent import create_agents
        agents = create_agents(room)
        print(f"[kgd-debug] {agents=}")
    except ModuleNotFoundError:
        agents = []

    inputs = {}

    # Initialize environment
    env = environment
    # Some environments return obs and info, some don't
    try:
        obs, info = env.reset()
    except TypeError:
        env.reset()
        obs = None

    # Initialize AI agents
    ai_agents = agents

    # Initialize variables
    step_count = 0
    terminated = False 
    truncated = False
    reward = 0
    # Get initial actions for agents
    actions = {}

    while not termination_condition(terminated, truncated):
        start_time = time.time()
        
        # Send data to server
        step_count = send_message(websocket, room, env, step_count, termination_condition, terminated, truncated, obs, actions, reward)
        # Wait for inputs from all users
        receive_message(websocket, room, users_needed, inputs)
        # Map inputs to actions or rewards
        if type == "action":
            actions = input_mapping(dict(inputs))
        else:
            reward = input_mapping(dict(inputs))
        
        # Get actions from AI agents
        if(not termination_condition(terminated, truncated)):
            generate_actions(ai_agents, obs, evaluate, actions)

        # Perform a step in the environment
        previous_obs = obs
        previous_reward = reward
        if type == "action":
            obs, reward, terminated, truncated, info = env.step(actions)
        else:
            obs, _, terminated, truncated, info = env.step(action_mapping(actions))

        if train and not evaluate:
            train_agents(ai_agents, previous_obs, actions, previous_reward, termination_condition(terminated, truncated), obs)

        loop_time = time.time() - start_time
        time.sleep(max(0, (1.0 / target_fps) - (loop_time + 0.001)))  # Maintain a minimum frame_rate
    # Send final message to server indicating termination
    send_message(websocket, room, env, step_count, termination_condition, terminated, truncated, obs, actions, reward)


def start_experiment(hostname, port, dirname, room, users_needed, type, target_fps, train, evaluate):
    try:
        with connect(f"ws://{hostname}:{port}/experiment/{dirname}/{'evaluate' if evaluate else 'run'}") as websocket:
            logging.info(f"Connected to experiment {dirname}")
            sys.path.append(f"experiments/{dirname}")
            logging.info(f"Starting experiment for room {room}")
            run_episode(websocket, room, users_needed, type, target_fps, train, evaluate)
        logging.info(f"Connection to {dirname} closed")
    except ConnectionRefusedError:
        logging.warning(f"Connection to {dirname} refused")



def get_all_directories(path):
    directories = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir != "__pycache__":
                directories.append(dir)
    return directories
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment runner.")
    parser.add_argument("--hostname", type=str, default="localhost", help="Hostname of the server")
    parser.add_argument("--port", type=int, default=8000, help="Port of the server")
    parser.add_argument("command", type=str, help="Command to run: 'runserver' or 'test'")
    args = parser.parse_args()
    hostname = args.hostname
    port = args.port

    sleeptime = 1.0
    dirs  = get_all_directories("experiments")
    while True:
        try:
            with connect(f"ws://{hostname}:{port}/experiment/queue") as websocket:
                sleeptime = 1.0
                logging.info(f"Connected to server's queue")
                while True:
                    websocket.send(json.dumps({"status": "idle"}))
                    message = json.loads(websocket.recv())
                    if 'experiment' in message.keys():
                        if message['experiment'] in dirs:
                            logging.info(f"Starting experiment {message['experiment']}")
                            p = Process(target=start_experiment, args=(hostname, port, message['experiment'], message['room'], message['users_needed'], message['type'], message['target_fps'], message['train'], message['evaluate'], ))
                            p.start()
                            p.join()
                        else:
                            logging.warning(f"Experiment {message['experiment']} not found in experiments folder")
                    time.sleep(1)
        except ConnectionClosedError:
            logging.warning(f"Connection to queue closed")
            continue
        except ConnectionRefusedError:
            logging.warning(f"Connection to queue refused")
            time.sleep(sleeptime)
            sleeptime = min(60.0, sleeptime * 2)
            continue
