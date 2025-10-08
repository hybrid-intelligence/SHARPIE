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
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(v) for v in data]
    else:
        return data


def run_entire_episode(websocket, room, users_needed):
    from environment import environment, termination_condition, input_mapping

    try:
        from agent import agents
    except ModuleNotFoundError:
        agents = []

    average_process_time = []
    average_transport_time = []

    env = environment
    env.reset()

    ai_agents = agents

    step_count = 0
    terminated = False 
    truncated = False
    actions = {}
    while not termination_condition(terminated, truncated):
        start_time = time.time()

        # Perform a step in the environment and get the rendered frame
        obs, reward, terminated, truncated, info = env.step(input_mapping(actions))
        frame = env.render()
        step_count += 1

        # Encode numpy frame as base64, makes it more compact for transfer
        _, buffer = cv2.imencode('.jpeg', frame.astype(np.uint8))
        image_base64 = base64.b64encode(buffer).decode('utf-8')

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
        end_time = time.time()
        average_process_time.append(end_time - start_time)
        
        start_transport = time.time()
        websocket.send(compressed_message)
        for i in range(users_needed):
            message = json.loads(websocket.recv())
            if 'message' in message.keys():
                logging.info(f"Message from room {room}: {message['message']}")
                if message['message'] == 'A user has disconnected':
                    return
            actions[message['session']['agent']] = message['actions']
        end_transport = time.time()
        average_transport_time.append(end_transport - start_transport)
        
        if(not termination_condition(terminated, truncated)):
            for ai_agent in ai_agents:
                actions[ai_agent.name] = ai_agent.sample(obs)


        loop_time = end_transport - start_time
        time.sleep(max(0, (1.0 / 24) - (loop_time + 0.001)))  # Maintain a minimum frame_rate
    
    logging.info(f"Experiment {room} over. Average process time: {int(sum(average_process_time) * 1000 / len(average_process_time))} ms. Average transport time: {int(sum(average_transport_time) * 1000 / len(average_transport_time))} ms.\n")



def start_experiment(dir, room, users_needed):
    try:
        with connect(f"ws://{hostname}:{port}/experiment/{dir}/run") as websocket:
            logging.info(f"Connected to experiment {dir}")
            sys.path.append(f"experiments/{dir}")
            logging.info(f"Starting experiment for room {room}")
            run_entire_episode(websocket, room, users_needed)
        logging.info(f"Connection to {dir} closed")
    except ConnectionRefusedError:
        logging.warning(f"Connection to {dir} refused")



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
                            p = Process(target=start_experiment, args=(message['experiment'], message['room'], message['users_needed']))
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