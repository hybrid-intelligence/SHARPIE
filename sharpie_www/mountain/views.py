from django.shortcuts import render
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

def index(request):
    env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=0.1)
    env = RecordVideo(env, video_folder="./static/mountain/", name_prefix="mountain_car", episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env, buffer_length=1)

    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    for _ in range(100):
        # this is where you would insert your policy
        action = env.action_space.sample()

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            break
    env.close()

    return render(request, "mountain/index.html", {"gym_load": "success", "steps": 100})
