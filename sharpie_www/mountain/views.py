
from django import shortcuts

import cv2
from PIL import Image
import ezpickle
import os
import gymnasium as gym
import numpy as np

from .q_table import *

import time

from rest_framework.response import Response
from rest_framework.decorators import api_view



def index_(request):
    return restart_(request)


def restart_(request):
    env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=0.1)
    discrete_observation = get_discrete_state(env.reset()[0])
    ezpickle.pickle_data(env, './static/mountain/env.pkl', overwrite=True)
    ezpickle.pickle_data(discrete_observation, './static/mountain/last_obs.pkl', overwrite=True)

    if not os.path.isfile('./static/mountain/q_table.pkl'):
        q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBSERVATION_SPACE_SIZE + [env.action_space.n]))
        ezpickle.pickle_data(q_table, './static/mountain/q_table.pkl', overwrite=True)
    #env.close()

    continue_(request)
    return shortcuts.render(request, "mountain/index.html")


@api_view(('GET',))
def continue_(request):
    print('Entering: ', time.time())
    env = ezpickle.unpickle_data('./static/mountain/env.pkl')
    discrete_observation = ezpickle.unpickle_data('./static/mountain/last_obs.pkl')
    q_table = ezpickle.unpickle_data('./static/mountain/q_table.pkl')

    if request.GET.get('left', False) == '1':
        action = 0
    elif request.GET.get('right', False) == '1':
        action = 2
    else:
        if request.GET.get('exploitation', 'false') == 'true' or np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_observation])
        else:
            action = np.random.randint(0, env.action_space.n)


    print('unzipping: ', time.time())
    observation, reward, terminated, truncated, info = env.step(action)
    print('step: ', time.time())
    discrete_observation = get_discrete_state(observation)
    ezpickle.pickle_data(env, './static/mountain/env.pkl', overwrite=True)
    ezpickle.pickle_data(discrete_observation, './static/mountain/last_obs.pkl', overwrite=True)
    print('zipping env-obs: ', time.time())

    if not (terminated or truncated or observation[0] > 0.55):
        # max q value for the next state calculated above
        max_future_q = np.max(q_table[discrete_observation])

        # q value for the current action and state
        current_q = q_table[discrete_observation + (action,)]

        new_q = (1 - LEARNING_RATE) * current_q + \
                LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # based on the new q, we update the current Q value
        q_table[discrete_observation + (action,)] = new_q

        # goal reached; reward = 0 and no more negative
    elif observation[0] > 0.55:
        q_table[discrete_observation + (action,)] = 100

    ezpickle.pickle_data(q_table, './static/mountain/q_table.pkl', overwrite=True)
    print('zipping agent: ', time.time())

    #img = env.render()
    print('rendering: ', time.time())
    #im = Image.fromarray(img)
    #print(time.time())
    #im.save('./static/mountain/step.png')
    #cv2.imwrite('./static/mountain/step.png', img)
    print('saving: ', time.time())

    if terminated or truncated or observation[0] > 0.55:
        env.close()
    data = {"is_terminated" : (1 if (terminated or truncated or observation[0] > 0.55) else 0)}
    return Response(data)