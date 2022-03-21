import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import gym
import numpy as np
from agent import Agent
from tqdm import tqdm

gamma = 0.99
learning_rate = 0.0005
memory_size = 50000
batch_size = 32
epsilon = 1.0
eps_decay = 0.999
eps_min = 0.05
episodes = 5000
target_update = 10

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    env.seed(42)
    agent = Agent(obs_dim=env.observation_space.shape,
            action_dim=env.action_space.shape[0], gamma=gamma,
            lr=learning_rate, eps=epsilon, eps_decay=eps_decay, eps_min=eps_min,
            env=env, memory_size=memory_size, batch_size=batch_size, dir='Modell')

    best_score = env.reward_range[0]
    max_steps = env._max_episode_steps
    total_steps = 0
    scores = []
    steps = []
    verbose = False
    progress = tqdm(range(episodes), desc='Training', unit=' episode')
    for i in progress:
        observation = env.reset()
        observation = observation.reshape(1, -1)
        done = False
        score = 0
        episode_steps = 0
        for step in range(max_steps):
            action = agent.get_action(observation)
            disc_action = agent.disc_actions(action)

            next_observation, reward, done, info = env.step(disc_action)
            next_observation = next_observation.reshape(1, -1)

            score += reward
            episode_steps += 1
            total_steps += 1

            agent.add_memory(observation, disc_action, reward, next_observation, done)
            observation = next_observation
            if done:
                break

        if (i % target_update) == 0:
            agent.q_target = agent.update(agent.q, agent.q_target)
            print('Updated')

        agent.learn()
        agent.greedy()
        scores.append(score)
        steps.append(episode_steps)
        average_score = np.mean(scores[-100:])

        print('\n')

        if average_score > best_score:
            best_score = average_score
            agent.save()
        if i % 20:
            np.save("reward.npy", np.array(scores))
            np.save("steps.npy", np.array(steps))
      
        print('Score %.1f' % score, 'Average Score %1f' % average_score, 'Steps %d' % episode_steps,
         'Total Steps %d' % total_steps, 'Epsilon %3f' % agent.epsilon)
        if average_score > 300:
            break
    np.save("reward.npy", np.array(scores))
    np.save("steps.npy", np.array(steps))