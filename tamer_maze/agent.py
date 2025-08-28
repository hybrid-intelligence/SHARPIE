import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
import gymnasium as gym
import pygame
# die Meinung, erledigen,  die Erfahrung, achten, versuchen, das Ziel, der Termin, schaffen, die Gewohnheit, der Alltag
# Urlaub

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


# Im Alltag, habe ich kaum Zeit zum Lezen
# Sein Alltag besteht aus Arbeit und Schlaf
# Der Alltag kan manchmal stressig sein
# Sie sucht im Alltag, kleine Freude
# Nach dem Urlaub, beginnt wieder  der Alltag 

# Rouchen ist eine schlechte Gewohnheit
# Die Kinderen lernen schnell neue Gewohnheiten 
# Man sollte sich gute Gewohnheiten aneignen
# Die Deutschen haben die Gewohnheit es, pünktlisch zu sein

# ich habe viel zu erledigen
# Kannst du den Einkauf erledigen?

class RewardModel(nn.Module): # given state and possible actions, this model needs to determine the reward for each of the actions
    def __init__(self, n_observations, n_actions, hidden_size=128):
        super(RewardModel, self).__init__()
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            'demo_ratio': 0.3,
            'hidden_size': 128,
            'grad_norm': 1.0
        }
        
        # Update with user config
        if config:
            self.config.update(config)
        
        # Initialize networks
        self.reward_net = RewardModel(state_size, action_size, self.config['hidden_size']).to(self.device)
        self.policy_net = DQN(state_size, action_size, self.config['hidden_size'])
        self.target_net = DQN(state_size, action_size, self.config['hidden_size'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=self.config['learning_rate'])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config['learning_rate'])
        
        # Initialize replay memories
        self.memory = ReplayMemory(self.config['memory_size'])

        # Training variables
        self.epsilon = self.config['epsilon']
        self.steps_done = 0
        self.training_losses = []
        
        # to device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        print(f"DQN Agent initialized on device: {self.device}")
    
    def get_debug_param(self):
        return self.config 
    
    # changed for TAMER implementation
    def select_action(self, state, use_epsilon=True):
        """Select action using TAMER algorithm"""
        self.reward_net.eval()
        # get the feature vector
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            rew_values = self.reward_net(state_tensor)
            return torch.argmax(rew_values, dim=-1).item()

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
        
        self.memory.push(state, action, reward, next_state, done)
    

    def train_rewardmodel(self, state, action_taken, human_reward):
        self.reward_net.train()
        device = self.device

        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        elif isinstance(state, torch.Tensor):
            state_tensor = state.float().unsqueeze(0).to(device)
        else:
            state_tensor = torch.tensor([state], dtype=torch.float32).to(device)

        action_tensor = torch.tensor([action_taken], dtype=torch.long, device=device)  # shape [1]
        human_reward_tensor = torch.tensor([human_reward], dtype=torch.float32, device=device)  # shape [1]

        predicted_rewards = self.reward_net(state_tensor) 
        predicted_reward_for_action = predicted_rewards.gather(1, action_tensor.unsqueeze(1)).squeeze(1)  # shape [1]

        loss = F.mse_loss(predicted_reward_for_action, human_reward_tensor)

        # backprop
        self.reward_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), self.config['grad_norm'])
        self.reward_optimizer.step()

        return loss.item()

    
    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < self.config['batch_size']:
            return None
        
        # Sample from both regular memory and demo memory
        #demo_batch_size = int(self.config['batch_size'] * self.config['demo_ratio']) if len(self.demo_memory) > 0 else 0
        regular_batch_size = self.config['batch_size']
        
        transitions = []
        if regular_batch_size > 0:
            transitions.extend(self.memory.sample(regular_batch_size))
        
        if not transitions:
            return None
        
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        
        # Handle next states
        non_terminal_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool).to(self.device)
        non_terminal_next_states = torch.FloatTensor(np.array([s for s in batch.next_state if s is not None])).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

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
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
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
        
        # Update target network periodically
        if self.steps_done % self.config['target_update_freq'] == 0:
            self.update_target_network()
        
        # Perform optimization
        loss = self.optimize_model()
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
            'replay_memory_capacity': self.config['memory_size'],
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