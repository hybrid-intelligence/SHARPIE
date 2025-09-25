#!/usr/bin/env python

"""Client using the asyncio API."""

import numpy as np
import base64
import cv2
import time
import pickle
import json
import gzip

from settings import environment, agents, termination_condition, input_mapping, user_per_experiment

from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosedError



def run_entire_episode(websocket, room):
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
            "observations": obs.tolist() if isinstance(obs, np.ndarray) else obs,
            "rewards": reward,
            "actions": actions,
            "image": image_base64
        }
        compressed_message = gzip.compress(json.dumps(group_message).encode('utf-8'))
        end_time = time.time()
        average_process_time.append(end_time - start_time)
        
        start_transport = time.time()
        websocket.send(compressed_message)
        for i in range(user_per_experiment):
            message = json.loads(websocket.recv())
            actions[message['session']['agent']] = message['actions']
        end_transport = time.time()
        average_transport_time.append(end_transport - start_transport)
        
        if(not termination_condition(terminated, truncated)):
            for ai_agent in ai_agents:
                actions[ai_agent.name] = ai_agent.sample(obs)


        loop_time = end_transport - start_time
        time.sleep(max(0, (1.0 / 24) - (loop_time + 0.001)))  # Maintain a minimum frame_rate
    
    print(f"Experiment {room} over. Average process time: {int(sum(average_process_time) * 1000 / len(average_process_time))} ms. Average transport time: {int(sum(average_transport_time) * 1000 / len(average_transport_time))} ms")



def start():
    with connect("ws://localhost:8000/experiment/run") as websocket:
        print("Connected to server")
        while True:
            print(f"Waiting for all users to connect")
            for i in range(user_per_experiment):
                message = json.loads(websocket.recv())
            print(f"Starting experiment for room {message['room']}")
            run_entire_episode(websocket, message['room'])


if __name__ == "__main__":
    while True:
        try:
            start()
        except ConnectionClosedError:
            print("Connection closed")
            continue
        except ConnectionRefusedError:
            print("Connection refused")
            time.sleep(1.0)
            continue