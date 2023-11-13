import gymnasium as gym
import os
import matplotlib.pyplot as plt
from ddqn_torch import DoubleQAgent
import numpy as np
import tqdm
import time

model = 'ddqn_lunar_lander/stats/m6.h5'

# m6;libT algDDQN mem200000 2NN(512,512) bat128 epi1500 lea4 rep100
def train_agent(n_episodes=1500,
                load_model=None,
                lr = 0.0001,
                epsilon = 1.0,
                epsilon_end = 0.01):
    LEARN_EVERY = 4
    
    print("Training a DDQN agent on {} episodes. Pretrained model = {}".format(n_episodes,load_model))
    env = gym.make(
            "LunarLander-v2",
            continuous = False,
            gravity = -10.0,
            enable_wind = False,
            wind_power = 15.0,
            turbulence_power = 1.5,
        )
    agent = DoubleQAgent(gamma=0.99,
                         epsilon=epsilon,
                         epsilon_dec=0.995,
                         lr=lr,
                         mem_size=200000,
                         batch_size=128,
                         epsilon_end=epsilon_end)
    if load_model:
        agent.load_saved_model("models/"+load_model)
        
        
    start_states = []
    scores = []
    eps_history = []
    start = time.time()
    best_avg = -1000
    
    for i in range(n_episodes):
        terminated = False
        truncated = False
        score = 0
        state = env.reset()[0]
        start_states.append(state)
        steps = 0
        while not (terminated or truncated):
            
            action = agent.choose_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            agent.save(state, action, reward, new_state, terminated)
            state = new_state
            if steps > 0 and steps % LEARN_EVERY == 0:
                agent.learn()
            steps += 1
            score += reward
            
        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[max(0, i-100):])

        if (i+1) % 10 == 0 and i > 0:
            # Report expected time to finish the training
            print('Episode {} in {:.2f} min. Expected total time for {} episodes: {:.0f} min. [{:.2f}/{:.2f}]'.format((i+1), 
                                                                                                                      (time.time() - start)/60, 
                                                                                                                      n_episodes, 
                                                                                                                      (((time.time() - start)/i)*n_episodes)/60, 
                                                                                                                      np.mean(scores[i-10:]), 
                                                                                                                      avg_score))
        
        if avg_score > best_avg and i > 100:
            best_avg = avg_score
            print(">>> NEW BEST AVERAGE:",best_avg)
            agent.save_model(f'models/ddqn_torch_model_{round(best_avg,2)}.h5')
      
    return agent, scores, start_states

def test(name, atype='single', test_games = 10):
    env = gym.make(
        "LunarLander-v2",
        continuous = False,
        gravity = -10.0,
        enable_wind = False,
        wind_power = 15.0,
        turbulence_power = 1.5,
    )
    rand_action_func = lambda x : np.exp(x/100 - 7)
    
    agent = DoubleQAgent(gamma=0.99, epsilon=0.0, lr=0.0005, mem_size=200000, batch_size=64, epsilon_end=0.01)
    agent.load_saved_model(name)
    state, _ = env.reset(seed=12)
    
    start_states = []
    total_rewards = []
    total_frames = []
    for _ in tqdm.trange(test_games):
        rewards = []
        terminated = False
        truncated = False
        state, _ = env.reset()
        start_states.append(state)
        frames = 0
        while not (terminated or truncated):
            if np.random.random() < rand_action_func(frames):
                action = 0
            else:
                action = agent.choose_action(state)
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state
            rewards.append(reward)
            frames += 1
        total_frames.append(frames)
        total_rewards.append(sum(rewards))
            
    return start_states, total_rewards, total_frames
            
train_agent()
# start_states, rewards, frames = test(model, test_games=2000)

# Apply PCA on the start states and plot the first two components
# Use the rewards and frames as the color of the points respectively in two different plots

# plt.plot(rewards, frames, 'o')
# plt.xlabel('rewards')
# plt.ylabel('frames')
# plt.savefig('rewardsXframes2.png')
