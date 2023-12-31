import gymnasium as gym
import os
import matplotlib.pyplot as plt
from ddqn_torch import DoubleQAgent
import numpy as np
import tqdm
import time
from torch.optim.lr_scheduler import LambdaLR

model = 'ddqn_lunar_lander/stats/m6.h5'

# m6;libT algDDQN mem200000 2NN(512,512) bat128 epi1500 lea4 rep100
def train_agent(n_episodes=100000,
                load_model=None,
                lr = 1e-4,
                epsilon = 1.0,
                epsilon_end = 0.01,
                save_path = "models",
                hidden_size=512,
                gamma=0.99):
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
    agent = DoubleQAgent(gamma=gamma,
                         epsilon=epsilon,
                         epsilon_dec=0.995,
                         lr=lr,
                         mem_size=200000,
                         batch_size=128,
                         epsilon_end=epsilon_end,
                         hidden_size = hidden_size)
    
    if load_model:
        agent.load_saved_model(load_model)
        
        
    start_states = []
    scores = []
    eps_history = []
    start = time.time()
    best_avg = -1000
    # update_at = updater_gen.__next__()
    
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
        
        if avg_score > best_avg and i > 100 and avg_score > 285:
            best_avg = avg_score
            print(">>> NEW BEST AVERAGE:",best_avg)
            if not os.path.exists(f'lunar-lander/ddqn_lunar_lander/saved_models/{save_path}'):
                os.makedirs(f'lunar-lander/ddqn_lunar_lander/saved_models/{save_path}')
            agent.save_model(f'lunar-lander/ddqn_lunar_lander/saved_models/{save_path}/ddqn_{int(best_avg)}')
      
    return agent, scores, start_states

def test(name, test_games = 10, hidden_size=512):
    env = gym.make(
        "LunarLander-v2",
        continuous = False,
        gravity = -10.0,
        enable_wind = False,
        wind_power = 15.0,
        turbulence_power = 1.5,
    )

    agent = DoubleQAgent(gamma=0.99, epsilon=0.0, lr=0.0005, mem_size=200000, batch_size=64, epsilon_end=0.01, hidden_size=hidden_size)
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
            action = agent.choose_action(state)
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state
            rewards.append(reward)
            frames += 1
        total_frames.append(frames)
        total_rewards.append(sum(rewards))
            
    return start_states, total_rewards, total_frames
            
def ensemble_test(models, test_games = 500, hidden_size = 512):
    env = gym.make(
        "LunarLander-v2",
        continuous = False,
        gravity = -10.0,
        enable_wind = False,
        wind_power = 15.0,
        turbulence_power = 1.5,
    )

    agents = []
    for model in models:
        agent = DoubleQAgent(gamma=0.99, epsilon=0.0, lr=0.0005, mem_size=200000, batch_size=64, epsilon_end=0.01, hidden_size = hidden_size)
        agent.load_saved_model(model)
        agents.append(agent)
        
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

            actions = []
            for agent in agents:
                action = agent.choose_action(state)
                actions.append(action)
            # Majority vote
            action = max(set(actions), key = actions.count)
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state
            rewards.append(reward)
            frames += 1
        total_frames.append(frames)
        total_rewards.append(sum(rewards))
            
    return start_states, total_rewards, total_frames
                

def plot_test(test, models, test_games = 500, plot_surfix = "", hidden_size=512):
    _, rewards, frames = test(models, test_games = test_games, hidden_size = hidden_size)
    
    plt.plot(rewards, frames, 'o')
    plt.legend()
    plt.xlabel('rewards')
    plt.ylabel('frames')
    plt.savefig(f'rewardsXframes_{plot_surfix}.png')
    plt.xlim(-600,400)
    plt.ylim(0,1000)
    plt.clf()
    print("Average reward:",np.mean(rewards), f"({plot_surfix})")
    

def batch_test(models, test_games = 500, plot_surfix = "", hidden_size = 512):
    
    for model in models:
        _, rewards, frames = test(model, test_games = test_games, hidden_size = hidden_size)
        print("Average reward:",np.mean(rewards), f"({model[-10:-3]})")
        plt.plot(rewards, frames, 'o', label = model)
    plt.legend()
    plt.xlabel('rewards')
    plt.ylabel('frames')
    plt.savefig(f'rewardsXframes_{plot_surfix}.png')
    plt.show()
    
save_path = "model_1028_finetune2"
model = f"lunar-lander/ddqn_lunar_lander/saved_models/model_1028_finetune2/ddqn_296.h5"
model_t = f"lunar-lander/ddqn_lunar_lander/saved_models/model_1028_finetune/ddqn_296_target.h5"
train_agent(load_model = model, save_path = save_path,
            hidden_size=1028, 
            lr = 1e-4, 
            epsilon=0.005, 
            epsilon_end = 0.005, 
            n_episodes=10000,
            gamma=0.997)


model1 = f"lunar-lander/ddqn_lunar_lander/saved_models/model_1028_finetune/ddqn_296.h5"
model2 = f"lunar-lander/ddqn_lunar_lander/saved_models/model_1028_finetune2/ddqn_296.h5"
batch_test([model1,model2], test_games = 500, plot_surfix = "295", hidden_size=1028)
# plot_test(test, model2, test_games = 500, plot_surfix = "m2")
# plot_test(test, model3, test_games = 500, plot_surfix = "m3")
# plot_test(test, model4, test_games = 500, plot_surfix = "m4")

