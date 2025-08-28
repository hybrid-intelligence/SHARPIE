import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
import gymnasium as gym
import pygame


# Define the transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    """Experience replay buffer for storing and sampling transitions"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DemoMemory:
    """Separate memory for human demonstrations"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a demonstration transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of demonstration transitions"""
        if len(self.memory) == 0:
            return []
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """Deep Q-Network architecture"""
    def __init__(self, n_observations, n_actions, hidden_size=128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        return self.layer3(x)

class DQNAgent:
    """DQN Agent with human demonstration support"""
    
    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size
        
        # Default hyperparameters
        self.config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon': 0.5,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': 32,
            'target_update_freq': 100,
            'memory_size': 10000,
            'demo_memory_size': 1000,
            'demo_ratio': 0.3,
            'hidden_size': 128,
            'grad_norm' : 1.0
        }
        
        # Update with user config
        if config:
            self.config.update(config)
        
        # Initialize networks
        print("state_size: ", state_size)
        self.policy_net = DQN(state_size, action_size, self.config['hidden_size'])
        self.target_net = DQN(state_size, action_size, self.config['hidden_size'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config['learning_rate'])
        
        # Initialize replay memories
        self.memory = ReplayMemory(self.config['memory_size'])
        self.demo_memory = DemoMemory(self.config['demo_memory_size'])
        
        # Training variables
        self.epsilon = self.config['epsilon']
        self.steps_done = 0
        self.training_losses = []
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        print(f"DQN Agent initialized on device: {self.device}")

    def get_debug_param(self):
        return self.config

    def select_action(self, state, use_epsilon=True):
        """Select action using epsilon-greedy policy"""
        if use_epsilon and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()

    def get_q_values(self, state):
        """Get Q-values for a given state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy().flatten()

    def store_transition(self, state, action, reward, next_state, done, is_demo=False):
        """Store a transition in the appropriate memory"""
        state = np.array(state).flatten()
        next_state = next_state if not done else None
        
        if is_demo:
            self.demo_memory.push(state, action, reward, next_state, done)
        else:
            self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < self.config['batch_size']:
            return None
        
        # Sample from both regular memory and demo memory
        demo_batch_size = int(self.config['batch_size'] * self.config['demo_ratio']) if len(self.demo_memory) > 0 else 0
        demo_batch_size = min(demo_batch_size, len(self.demo_memory))
        
        regular_batch_size = self.config['batch_size'] - demo_batch_size
        regular_batch_size = min(regular_batch_size, len(self.memory))

        transitions = []
        if regular_batch_size > 0:
            transitions.extend(self.memory.sample(regular_batch_size))
        if demo_batch_size > 0:
            transitions.extend(self.demo_memory.sample(demo_batch_size))
        
        if not transitions:
            return None
        
        batch = Transition(*zip(*transitions))

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        

        # Handle next states
        non_terminal_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool).to(self.device)
        non_terminal_next_states = torch.FloatTensor(
            np.vstack([np.array(s) for s in batch.next_state if s is not None])
        ).to(self.device)

        print("state_batch.shape: ", state_batch.shape)
        print("action_batch.shape: ", action_batch.shape)
        print("reward_batch.shape: ", reward_batch.shape)
        print("non_terminal_mask: ", non_terminal_mask.shape)
        print("non_terminal_next_states: ", non_terminal_next_states.shape)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        print("state_action_values.shape: ", state_action_values.shape)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.config['batch_size']).to(self.device)
        if len(non_terminal_next_states) > 0:
            with torch.no_grad():
                next_state_values[non_terminal_mask] = self.target_net(non_terminal_next_states).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.config['gamma'] * next_state_values)

        # Compute loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config['grad_norm'])
        self.optimizer.step()
        
        # Store loss for tracking
        self.training_losses.append(loss.item())
        
        return loss.item()

    def update_target_network(self):
        """Update target network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay epsilon for exploration"""
        self.epsilon = max(self.config['epsilon_min'], self.config['epsilon_decay'] * self.epsilon)

    def step(self):
        """Perform one training step"""
        self.steps_done += 1
        print("d1")
        # Update target network periodically
        if self.steps_done % self.config['target_update_freq'] == 0:
            self.update_target_network()
        print("d2")
        # Perform optimization
        loss = self.optimize_model()
        print("d3")
        return loss

    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'config': self.config
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.config.update(checkpoint['config'])
        print(f"Model loaded from {filepath}")

    def get_memory_stats(self):
        """Get statistics about memory usage"""
        return {
            'replay_memory_size': len(self.memory),
            'demo_memory_size': len(self.demo_memory),
            'replay_memory_capacity': self.config['memory_size'],
            'demo_memory_capacity': self.config['demo_memory_size']
        }

    def get_training_stats(self):
        """Get training statistics"""
        return {
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0,
            'total_losses': len(self.training_losses)
        }

    def reset_epsilon(self, epsilon=None):
        """Reset epsilon to initial value or specified value"""
        self.epsilon = epsilon if epsilon is not None else self.config['epsilon']

    def set_eval_mode(self):
        """Set networks to evaluation mode"""
        self.policy_net.eval()
        self.target_net.eval()

    def set_train_mode(self):
        """Set networks to training mode"""
        self.policy_net.train()
        self.target_net.eval()  # Target network always in eval mode