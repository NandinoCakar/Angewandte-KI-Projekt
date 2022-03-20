import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.chdir("C:/Users/nandi/OneDrive - Siemens Energy/Master/BipedalWalker/SAC Abgabe")
import gym
import numpy as np
from agent import Agent
from tqdm import tqdm


gamma = 0.99
tau = 0.02
learning_rate = 0.0003
memory_size = 1000000
batch_size = 256
start_steps = 10000


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    env.seed(42)
    agent = Agent(obs_dim=env.observation_space.shape,
            action_dim=env.action_space.shape[0], gamma=gamma,
            tau=tau, lr=learning_rate, env=env, memory_size=memory_size, 
            batch_size=batch_size, dir='walker 6')

    best_score = env.reward_range[0]
    max_steps = env._max_episode_steps
    total_steps = 0
    scores = []
    average_actor_loss = []
    average_q1_loss = []
    average_q2_loss = []
    steps = []
    alphas = []
    progress = tqdm(range(2000), desc='Training', unit=' episode')
    for i in progress:
        observation = env.reset()
        done = False
        score = 0
        episode_steps = 0
        actor_loss = []
        q1_loss = []
        q2_loss = []
        for step in range(max_steps):

            if start_steps > total_steps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.actor.choose_action(observation)  # Sample action from policy
            

            next_observation, reward, done, info = env.step(action)
            next_observation = next_observation.astype(np.float32)
            score += reward
            episode_steps += 1
            total_steps += 1

            done = 1 if done is True else 0

            agent.add_memory(observation, action, reward, next_observation, done)
            observation = next_observation
            if total_steps > batch_size:
                agent.learn()
                actor_loss.append(agent.actor_loss)
                q1_loss.append(agent.q1_loss)
                q2_loss.append(agent.q2_loss)
            if done:
                break
        scores.append(score)
        steps.append(episode_steps)
        alphas.append(agent.alpha)
        if total_steps > batch_size:
            actor_loss = np.mean(actor_loss)
            q1_loss = np.mean(q1_loss)
            q2_loss = np.mean(q2_loss)
            average_actor_loss.append(actor_loss)
            average_q1_loss.append(q1_loss)
            average_q2_loss.append(q2_loss)

        average_score = np.mean(scores[-100:])
        print('\n')
        if average_score > best_score:
            best_score = average_score
            agent.save()
        if i % 20:
            np.save("actor_loss.npy", np.array(average_actor_loss))
            np.save("critic1_loss.npy", np.array(average_q1_loss))
            np.save("critic2_loss.npy", np.array(average_q2_loss))
            np.save("reward.npy", np.array(scores))
            np.save("steps.npy", np.array(steps))
            np.save("alpha.npy", np.array(alphas))
        if total_steps > batch_size:
            print('Actor Loss:  %.4f ' % actor_loss, 'Q1 Loss: %.4f' % q1_loss,'Q2 Loss: %.4f' % q2_loss, 'Alpha: %.4f' % agent.alpha)
            print('Average Actor Loss: %.4f' % np.mean(average_actor_loss[-100:]), 'Average Q1 Loss: %.4f'  % np.mean(average_q1_loss[-100:]),
                 'Average Q2 Loss: %.4f'  % np.mean(average_q2_loss[-100:]))
        print('Score %.1f' % score, 'Average Score %.1f' % average_score, 'Steps %.1f' % episode_steps, 'Total Steps %.1f' % total_steps)
        if average_score > 300:
            break
    np.save("actor_loss.npy", np.array(average_actor_loss))
    np.save("critic1_loss.npy", np.array(average_q1_loss))
    np.save("critic2_loss.npy", np.array(average_q2_loss))
    np.save("reward.npy", np.array(scores))
    np.save("steps.npy", np.array(steps))
    np.save("alpha.npy", np.array(alphas))