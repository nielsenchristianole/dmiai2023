import gymnasium as gym
import os
import matplotlib.pyplot as plt
from ddqn_torch import DoubleQAgent
import numpy as np
import tqdm

model = 'lunar-lander-ddqn/stats/m6.h5'

def test(name, atype='single', test_games = 10):
    env = gym.make(
        "LunarLander-v2",
        continuous = False,
        gravity = -10.0,
        enable_wind = False,
        wind_power = 15.0,
        turbulence_power = 1.5,
    )
    agent = DoubleQAgent(gamma=0.99, epsilon=0.0, lr=0.0005, mem_size=200000, batch_size=64, epsilon_end=0.01)
    agent.load_saved_model(name)
    state, info = env.reset(seed=12)
    
    sums = []
    for _ in range(test_games):
        terminated = False
        truncated = False
        state, info = env.reset()
        while not (terminated or truncated):
            print(type(state))
            action = agent.choose_action(state)
            print(">>action:",type(action.item()),action.item())
            new_state, reward, terminated, truncated, info = env.step(action)
            state = new_state
            

test(model, test_games=20)