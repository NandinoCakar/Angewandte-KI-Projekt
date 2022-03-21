import gym, time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agent import Agent

if __name__ == "__main__":

    gamma = 0.99
    learning_rate = 0.0005
    memory_size = 50000
    batch_size = 32
    epsilon = 0
    eps_decay = 0.999
    eps_min = 0.05
    test_episodes = 5
    max_steps = 1600


    env = gym.make('BipedalWalker-v3')
    env = env.unwrapped

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = Agent(obs_dim=env.observation_space.shape,
                      action_dim=env.action_space.shape[0], gamma=gamma,
                      lr=learning_rate, eps=epsilon, eps_decay=eps_decay, eps_min=eps_min,
                      env=env, memory_size=memory_size, batch_size=batch_size, dir='Modell')

    t0 = time.time()
    agent.load()
    rewards = []
    epsilon_history = []
    steps = 0

    for epoch in range(test_episodes):
        done = False
        state = env.reset()
        state = state.reshape(1, -1)
        start = time.time()
        total_reward = 0
        steps = 0
        for step in range(max_steps):
            steps += 1
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape(1, -1)
            total_reward += reward
            env.render()

            state = next_state
            if done:
                break
        print(
            'Testing  | Episode: {}/{}  | Reward: {:.4f} | Steps: {:.4f} | Running Time: {:.4f}'.format(
                epoch + 1, test_episodes, reward, steps,
                time.time() - t0
            )
        )
