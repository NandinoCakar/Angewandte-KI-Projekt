import numpy as np
np.random.seed(42)

class Memory:
    def __init__(self, memory_size, obs_dim, action_dim):
        self.memory_size = memory_size
        self.memory_counter = 0 
        self.state_memory = np.zeros((self.memory_size, 1, *obs_dim))
        self.next_state_memory = np.zeros((self.memory_size, 1, *obs_dim))
        self.action_memory = np.zeros((self.memory_size, action_dim))
        self.reward_memory = np.zeros(self.memory_size)
        self.done_memory = np.zeros(self.memory_size, dtype=np.bool)

    # Stores the transition in the memory
    def add_memory(self, state, action, reward, next_state, done):
        index = self.memory_counter % self.memory_size
        self.memory_counter += 1
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = done
    
    # Get random batch from the memory
    def sample_batch(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        batch_index = np.random.choice(max_memory, batch_size)

        states = self.state_memory[batch_index]
        actions = self.action_memory[batch_index]
        rewards = self.reward_memory[batch_index]
        next_states = self.next_state_memory[batch_index]
        dones = self.done_memory[batch_index]

        return states, actions, rewards, next_states, dones