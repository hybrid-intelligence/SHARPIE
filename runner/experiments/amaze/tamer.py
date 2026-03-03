# This code was taken from https://github.com/benibienz/TAMER/tree/master

import os
import pickle
from pathlib import Path

import numpy as np
from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
import gymnasium as gym

MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')


class SGDFunctionApproximator:
    """ SGD function approximator with RBF preprocessing. """
    def __init__(self, env):
        # Feature preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(100000)], dtype='float64'
        )
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurized represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = pipeline.FeatureUnion(
            [
                ('rbf1', RBFSampler(gamma=5.0, n_components=100, random_state=1)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=100, random_state=1)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=100, random_state=1)),
                ('rbf4', RBFSampler(gamma=0.5, n_components=100, random_state=1)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))

        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(alpha=0.1, learning_rate='constant', random_state=1)
            model.partial_fit([self.featurize_state(env.reset()[0])], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        features = self.featurize_state(state)
        if not action:
            return [m.predict([features])[0] for m in self.models]
        else:
            return self.models[action].predict([features])[0]

    def update(self, state, action, td_target):
        features = self.featurize_state(state)
        self.models[action].partial_fit([features], [td_target])

    def featurize_state(self, state):
        """ Returns the featurized representation for a state. """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

class TabularLearner:
    """Only works for discrete state-action environments."""
    def __init__(self, env, learning_rate = 1/3):
        # used for sampling actions
        self.h_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.learning_rate = learning_rate

    def predict(self, state):
        state = state[0]
        return self.h_table[state]
    
    def update(self, state, action, target):
        state = state[0]
        error = self.h_table[state, action] - target
        self.h_table[state,action] += self.learning_rate * -error

class Tamer:
    """
    QLearning Agent adapted to TAMER using steps from:
    http://www.cs.utexas.edu/users/bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html
    """
    def __init__(
        self,
        environment,
        model_file_to_load=None  # filename of pretrained model
    ):
        self.env = environment
        # init model
        if model_file_to_load is not None and os.path.isfile(MODELS_DIR.joinpath(model_file_to_load+'.p')):
            self.load_model(filename=model_file_to_load)
        else:
            self.H = SGDFunctionApproximator(self.env)

    def argmax_random(self, x):
        x = np.array(x)
        max = x.max()
        maxes = np.flatnonzero(x == max)
        return np.random.choice(maxes)

    def act(self, state, epsilon=0.0):
        if np.random.random() < 1 - epsilon:
            print(self.predict(state))
            return self.argmax_random(self.predict(state))
        else:
            return np.random.randint(0, self.env.action_space.n)
 
    def predict(self, state):
        return self.H.predict(state)
    
    def train(self, state, action, td_target):
        self.H.update(state, action, td_target)

    def save_model(self, filename):
        """
        Save H or Q model to models dir
        Args:
            filename: name of pickled file
        """
        model = self.H
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, filename):
        """
        Load H or Q model from models dir
        Args:
            filename: name of pickled file
        """
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            self.H = pickle.load(f)
