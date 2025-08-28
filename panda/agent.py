# used a ddpg implementation, I forgot which one was it :(())
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import gymnasium as gym




class Actor(nn.Module):
    def __init__(self,state_size, action_size, hidden_size = 256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) # makes it continuous
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size = 256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, buffer_size = 100000):
        self.buffer = deque(maxlen=buffer_size)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class DDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor = 1e-4, lr_critic = 1e-3, gamma = 0.99, tau = 1e-3, device = 'cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        # networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.noise = OUNoise(action_dim)
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        #print("state.shape: ",state.shape)
        action = self.actor(state).squeeze(0).detach().cpu().numpy()
        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1) # we are dealing with spaces of (-1,1)
    
    def learn(self, batch_size = 64):
        if len(self.replay_buffer) < batch_size: # not enough space
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        #print("state", state)
        #print("next_state", next_state)
        if isinstance(state, dict):
            state = np.concatenate(state['observation'], state['desired_goal'])
        if isinstance(next_state, dict):
            next_state =  np.concatenate(next_state['observation'], next_state['desired_goal'])
        #print("next_state123", next_state)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device) # not sure about the unsqueezing here. 

        # update critic
        next_action = self.actor_target(next_state)
        next_q_value= self.critic_target(next_state, next_action)
        target_q_value = reward + (1-done)*self.gamma*next_q_value 
        current_q_value = self.critic(state, action)
        critic_loss = F.mse_loss(current_q_value, target_q_value.detach())
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        actor_loss = -self.critic(state, self.actor(state)).mean() # take mean tp get a single value across batches
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
    
    def soft_update(self, local_model, target_model):
        """Soft update target network parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

