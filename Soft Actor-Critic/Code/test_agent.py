import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.chdir("C:/Users/nandi/") # Change to your Folder
import gym
import numpy as np
import time
from agent import Agent
from tqdm import tqdm


gamma = 0.99
tau = 0.01
learning_rate = 0.0003
memory_size = 1000000
batch_size = 256
test_episode = 5

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    env.seed(42)
    agent = Agent(obs_dim=env.observation_space.shape,
            action_dim=env.action_space.shape[0], gamma=gamma,
            tau=tau, lr=learning_rate, env=env, memory_size=memory_size,
            batch_size=batch_size, dir='walker 6')

    agent.load()
    best_score = env.reward_range[0]
    max_steps = 1600
    t0 = time.time()
    for episode in range(test_episode):
        total_steps = 0
        score = 0
        episode_steps = 0
        observation = env.reset()
        done = False
        for step in range(max_steps):
            env.render()

            action = agent.actor.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            next_observation = next_observation.astype(np.float32)
            score += reward
            episode_steps += 1
            total_steps += 1

            observation = next_observation
            if done:
                break
        print(
            'Testing  | Episode: {}/{}  | Episode Reward: {:.4f} |  | Running Time: {:.4f}'.format(
                episode + 1, test_episode, score,
                time.time() - t0
            )
        )
