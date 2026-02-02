import numpy as np
from typing import Tuple, Callable, Optional
from collections import deque

class TAMERAgent:
    """
    TAMER agent implementation based on Algorithm 2 from Knox & Stone (2009).
    Learns a reward model from human evaluative feedback.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.01,
                 discount_factor: float = 0.99,
                 feedback_buffer_size: int = 1000):
        """
        Initialize TAMER agent.
        
        Args:
            state_dim: Dimensionality of state space
            action_dim: Number of discrete actions
            learning_rate: Learning rate for reward model updates
            discount_factor: Discount factor for future rewards
            feedback_buffer_size: Size of feedback history buffer
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Reward model: linear function approximator
        # Weights for [state_features, action_one_hot]
        self.reward_weights = np.random.randn(state_dim + action_dim) * 0.01
        
        # Feedback history for batch updates
        self.feedback_buffer = deque(maxlen=feedback_buffer_size)

        
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action greedily with respect to learned reward model.
        
        Args:
            state: Current state
            
        Returns:
            Selected action (greedy w.r.t. learned reward)
        """
        # Compute estimated reward for each action
        q_values = []
        for action in range(self.action_dim):
            state_action = self._encode_state_action(state, action)
            q_value = np.dot(state_action, self.reward_weights)
            q_values.append(q_value)
        
        return np.argmax(q_values)
    
    def update_reward_model(self, 
                           state: np.ndarray, 
                           action: int, 
                           human_feedback: float) -> None:
        """
        Update reward model weights based on human evaluative feedback.
        Implements Algorithm 2 from Knox & Stone (2009).
        
        Args:
            state: State where action was taken
            action: Action that was executed
            human_feedback: Human evaluative feedback signal
        """
        state_action = self._encode_state_action(state, action)
        
        # Current estimate
        current_estimate = np.dot(state_action, self.reward_weights)
        
        # Prediction error
        error = human_feedback - current_estimate
        
        # Update rule: w = w + α * (h - H(s,a)) * ∇H(s,a)
        # For linear model: ∇H(s,a) = φ(s,a)
        self.reward_weights += self.learning_rate * error * state_action
        
        # Store in buffer for potential batch updates
        self.feedback_buffer.append({
            'state': state.copy(),
            'action': action,
            'feedback': human_feedback,
            'estimate': current_estimate,
            'error': error
        })
    
    def batch_update(self, feedback_samples: list) -> float:
        """
        Perform batch update on multiple feedback samples.
        
        Args:
            feedback_samples: List of feedback dictionaries
            
        Returns:
            Mean squared error across samples
        """
        mse = 0.0
        for sample in feedback_samples:
            state_action = self._encode_state_action(sample['state'], sample['action'])
            current_estimate = np.dot(state_action, self.reward_weights)
            error = sample['feedback'] - current_estimate
            self.reward_weights += self.learning_rate * error * state_action
            mse += error ** 2
        
        return mse / len(feedback_samples) if feedback_samples else 0.0
    
    def get_reward_estimate(self, state: np.ndarray, action: int) -> float:
        """
        Get the learned reward estimate for a state-action pair.
        
        Args:
            state: State
            action: Action
            
        Returns:
            Estimated reward
        """
        state_action = self._encode_state_action(state, action)
        return np.dot(state_action, self.reward_weights)
    
    def save_model(self, filepath: str) -> None:
        """
        Save reward model weights to file.
        
        Args:
            filepath: Path to save the model
        """
        np.save(filepath, self.reward_weights)
        print("Stored TAMER model from:", filepath)
        
    def load_model(self, filepath: str) -> None:
        """
        Load reward model weights from file.
        
        Args:
            filepath: Path to load the model from
        """
        self.reward_weights = np.load(filepath)
        print("Loaded TAMER model from:", filepath)

    
    def _encode_state_action(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Encode state-action pair as feature vector.
        
        Args:
            state: State vector
            action: Action index
            
        Returns:
            Concatenated feature vector [state_features, action_one_hot]
        """
        action_vec = np.zeros(self.action_dim)
        action_vec[action] = 1.0
        return np.concatenate([state, action_vec])