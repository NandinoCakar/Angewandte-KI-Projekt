import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(42)  

class CriticNetwork(Model):
    def __init__(self, action_dim, neurons=256):
        super(CriticNetwork, self).__init__()
        self.action_dim = action_dim

        self.layer1 = Dense(neurons, activation='relu')
        self.layer2 = Dense(neurons, activation='relu')
        self.q = Dense(1, activation='linear')

    def call(self, state, action):
        state_action_value = self.layer1(tf.concat([state, action], axis=1))
        state_action_value = self.layer2(state_action_value)
        q = self.q(state_action_value)
        return q

class ActorNetwork(Model):
    def __init__(self, action_dim, action_limit, neurons=256, init_weight=3e-3):
        super(ActorNetwork, self).__init__()
        self.action_dim = action_dim
        self.noise = 1e-6
        self.action_limit = action_limit

        self.layer1 = Dense(neurons, activation='relu')
        self.layer2 = Dense(neurons, activation='relu')
        self.layer3 = Dense(neurons, activation='relu')
        self.mean = Dense(self.action_dim, activation='linear', bias_initializer=tf.random_uniform_initializer(-init_weight, init_weight))
        self.log_std = Dense(self.action_dim, activation='linear', bias_initializer=tf.random_uniform_initializer(-init_weight, init_weight))
    
    
    def call(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)

        mean = self.mean(x) 
        log_std = self.log_std(x)
        log_std = tf.clip_by_value(log_std, -20, 2)

        return mean, log_std


    def sample(self, state):
        mean, log_std = self.call(state)
        std = tf.math.exp(log_std) 

        normal = tfd.Normal(0, 1)
        z = normal.sample(tf.shape(mean))
        action_0 = tf.math.tanh(mean + std * z) # Reparameterization trick from the paper
        action = self.action_limit * action_0 # Normalize actions with enviroment action range (not necessary for Walker)
        log_pi = tfd.Normal(mean, std).log_prob(mean + std * z)
        log_pi -= tf.math.log(1.0 - action_0 ** 2 + self.noise) - np.log(self.action_limit) # Paper Appendix C
        log_pi = tf.reduce_sum(log_pi, axis=1)[:, np.newaxis] # Expand dim because reduce_sum reduces dim

        return action, log_pi

    def choose_action(self, state):
        state = tf.convert_to_tensor([state])
        mean, log_std = self.call(state)
        std = tf.math.exp(log_std)

        normal = tfd.Normal(0, 1)
        z = normal.sample(tf.shape(mean))
        action = self.action_limit * tf.math.tanh(mean + std * z)
        return action[0]
