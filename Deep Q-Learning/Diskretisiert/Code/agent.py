import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as k
from memory import Memory

class Agent():
    def __init__(self, obs_dim, action_dim, gamma,
                lr, eps, eps_decay, eps_min, env=None, memory_size=50000,
                batch_size=256, dir='dqn'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_limit = env.action_space.high
        self.gamma = gamma # Discount factor
        self.lr = lr # Learning rate for all networks
        self.epsilon = eps
        self.epsilon_decay = eps_decay
        self.epsilon_min = eps_min
        self.memory = Memory(memory_size, obs_dim, action_dim)
        self.batch_size = batch_size
        self.dir = dir
        self.disc_steps = 7


        # Q Network
        self.q = self.network(obs_dim[0], action_dim)

        # Target Network
        self.q_target = self.network(obs_dim[0], action_dim)


    def get_action(self, observation):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-1, 1, 4)
        action = self.q.predict(observation)
        return action[0]

    def disc_actions(self, action):
        rnd = (self.disc_steps-1) / 2
        disc_action = np.around(action * rnd)/ rnd
        return disc_action

    def network(self, obs_dim, action_dim):
        q_network = Sequential()
        q_network.add(Dense(64, activation='relu', input_dim=obs_dim))
        q_network.add(Dense(64, activation='relu'))
        q_network.add(Dense(action_dim, activation='linear'))
        q_network.compile(optimizer=Adam(lr=self.lr), loss='mse')
        return q_network

    def learn(self):
        if self.memory.memory_counter > self.batch_size:
            state, action, reward, next_state, done = self.memory.sample_batch(self.batch_size)
            for batch_index in range(self.batch_size):
                q_next = self.q_target.predict(next_state[batch_index])
                q_target = reward[batch_index] + self.gamma * (1-done[batch_index]) * np.max(q_next)
                q_target_old = self.q.predict(state[batch_index])
                q_target_old[0] = q_target
                self.q.fit(state[batch_index], q_target_old, verbose=0, epochs=1)
                k.clear_session()

    def update(self, q, q_target):
        for target_weight, q_weight in zip(q_target.trainable_weights, q.trainable_weights):
            target_weight.assign(q_weight)
        return q_target
            
    def greedy(self):
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
    
    def add_memory(self, state, action, reward, next_state, done):
        self.memory.add_memory(state, action, reward, next_state, done)

    def save(self):
        self.q.save_weights(self.dir + "/q.ckpt")
        print("Models saved")
    
    def load(self):
        self.q.load_weights(self.dir+"/q.ckpt")
        print("Models loaded")