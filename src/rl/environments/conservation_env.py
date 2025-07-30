"""Conservation Environment for RL training."""

import gym
from gym import spaces
import numpy as np

class ConservationEnvironment(gym.Env):
    def __init__(self, config):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,))
        self.action_space = spaces.Discrete(10)
        
    def reset(self):
        return np.random.uniform(-1, 1, 10)
    
    def step(self, action):
        next_state = np.random.uniform(-1, 1, 10)
        reward = np.random.random()
        done = np.random.random() < 0.1
        return next_state, reward, done, {}
