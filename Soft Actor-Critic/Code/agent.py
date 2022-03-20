import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from memory import Memory
from networks import ActorNetwork, CriticNetwork

class Agent():
    def __init__(self, obs_dim, action_dim, gamma,
                tau, lr, env=None, memory_size=1000000, 
                batch_size=256, dir='walker 4'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_limit = env.action_space.high
        self.gamma = gamma # Discount factor
        self.tau = tau # Interpolation factor for updating target networks
        self.lr = lr # Learning rate for all networks
        self.memory = Memory(memory_size, obs_dim, action_dim)
        self.batch_size = batch_size
        self.dir = dir
        self.log_alpha = tf.Variable(0, dtype=np.float32, name='log_alpha')
        # self.alpha = tf.math.exp(self.log_alpha)
        self.alpha = 0.2
        self.target_entropy = -1. * action_dim # -dim(actions)


        # Policy Network (Actor)
        self.actor = ActorNetwork(action_dim=action_dim, action_limit = self.action_limit)

        # 2 Critic Networks and 2 Target Networks
        self.critic1 = CriticNetwork(action_dim=action_dim)
        self.target1 = CriticNetwork(action_dim=action_dim)
        self.critic2 = CriticNetwork(action_dim=action_dim)
        self.target2 = CriticNetwork(action_dim=action_dim)

        # Initialize target network weights 
        # tau = 1 -> hard update
        self.target1 = self.soft_update(self.critic1, self.target1, 1)
        self.target2 = self.soft_update(self.critic2, self.target2, 1)


        self.critic1_opt = tf.optimizers.Adam(self.lr)
        self.critic2_opt = tf.optimizers.Adam(self.lr)
        self.actor_opt = tf.optimizers.Adam(self.lr)
        self.alpha_opt = tf.optimizers.Adam(self.lr) # optimizer for automatic entropy regularization
        



    # Update Target Network according to tau*target_weights + (1-tau)*critic_weights
    # Hard copy if tau=1
    def soft_update(self, critic, target, tau):
        for target_weight, critic_weight in zip(target.trainable_weights, critic.trainable_weights):
            target_weight.assign(target_weight * (1.0 - tau) + critic_weight * tau)
        return target
        

    def learn(self):
        state, action, reward, next_state, done = self.memory.sample_batch(self.batch_size)

        reward = reward[:, np.newaxis]  # Expand dim
        done = done[:, np.newaxis]

        reward = reward - np.mean(reward, axis=0) / (np.std(reward, axis=0) + 1e-6)  # Normalize reward with batch
        

        next_action, log_pi = self.actor.sample(next_state)
        target_value1 = self.target1(next_state, next_action)
        target_value2 = self.target2(next_state, next_action)
        target_value = tf.math.minimum(target_value1, target_value2) - self.alpha * log_pi # Entropy-regularized Bellman

        target_q = reward + self.gamma * (1-done) * target_value 

        # Training based on gradients
        self.train_critic1(target_q, state, action)
        self.train_critic2(target_q, state, action)
        log_pi = self.train_actor(state)
        
        # Auto regularization of entropy coefficient
        # self.auto_alpha(log_pi)

        # Update Target Networks
        self.target1 = self.soft_update(self.critic1, self.target1, self.tau)
        self.target2 = self.soft_update(self.critic2, self.target2, self.tau)

    def train_critic1(self, target_q, state, action):
        with tf.GradientTape() as tape_q1:
            q1 = self.critic1(state, action)
            self.q1_loss = tf.reduce_mean(tf.losses.mean_squared_error(q1, target_q))
        q1_gradient= tape_q1.gradient(self.q1_loss, self.critic1.trainable_weights)
        self.critic1_opt.apply_gradients(zip(q1_gradient, self.critic1.trainable_weights))

    def train_critic2(self, target_q, state, action):
        with tf.GradientTape() as tape_q2:
            q2 = self.critic2(state, action)
            self.q2_loss = tf.reduce_mean(tf.losses.mean_squared_error(q2, target_q))
        q2_gradient = tape_q2.gradient(self.q2_loss, self.critic2.trainable_weights)
        self.critic2_opt.apply_gradients(zip(q2_gradient, self.critic2.trainable_weights))
    
    def train_actor(self, state):
        with tf.GradientTape() as tape_actor:
            action, log_pi = self.actor.sample(state)
            q = tf.math.minimum(self.critic1(state, action), self.critic2(state, action))
            self.actor_loss = tf.reduce_mean(self.alpha * log_pi - q)
        actor_gradient = tape_actor.gradient(self.actor_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_gradient, self.actor.trainable_weights))
        return log_pi
    
    def auto_alpha(self, log_pi):
        with tf.GradientTape() as alpha_tape:
            alpha_loss = -tf.reduce_mean((self.log_alpha * (log_pi + self.target_entropy)))
        alpha_grad = alpha_tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_opt.apply_gradients(zip(alpha_grad, [self.log_alpha]))
        self.alpha = tf.math.exp(self.log_alpha)
 
    def add_memory(self, state, action, reward, next_state, done):
        self.memory.add_memory(state, action, reward, next_state, done)
    
    def save(self):
        self.actor.save_weights(self.dir + "/actor.ckpt")
        self.critic1.save_weights(self.dir + "/critic1.ckpt")
        self.critic2.save_weights(self.dir + "/critic2.ckpt")
        self.target1.save_weights(self.dir + "/target1.ckpt")
        self.target2.save_weights(self.dir + "/target2.ckpt")
        print("Models saved")
    
    def load(self):
        self.actor.load_weights(self.dir+"/actor.ckpt")
        self.critic1.load_weights(self.dir+"/critic1.ckpt")
        self.critic2.load_weights(self.dir+"/critic2.ckpt")
        self.target1.load_weights(self.dir+"/target1.ckpt")
        self.target2.load_weights(self.dir+"/target2.ckpt")
        print("Models loaded")