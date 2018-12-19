import random
from collections import namedtuple, deque

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, p):  # action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.p = p  # p is shorthand for Parameters
        self.memory = deque(maxlen=p.BUFFER_SIZE)
        self.experience = namedtuple("Experience",
                                     field_names=["prev_state", "state", "action", "reward", "next_state", "done"])

    def add(self, prev_state, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(prev_state, state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.p.BATCH_SIZE)

        prev_states = torch.from_numpy(np.vstack([e.prev_state for e in experiences if e is not None])).float().to(
            self.p.DEVICE)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.p.DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(
            self.p.DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(
            self.p.DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.p.DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.p.DEVICE)

        return (prev_states, states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
