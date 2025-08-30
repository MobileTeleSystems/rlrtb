import torch
import random
import numpy as np

from typing import Tuple
from collections import namedtuple, deque

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """
    The ReplayBuffer class implements a fixed-size experience replay buffer for reinforcement learning agents.

    This buffer stores experience tuples (state, action, reward, next_state, done) and provides functionality to
    randomly sample batches of experiences for training. Experiences are automatically evicted when the buffer reaches
    its maximum capacity to maintain temporal diversity in training samples.

    Parameters:
        buffer_size (int): maximum number of experience tuples the buffer can hold
        batch_size (int): number of experiences to sample per training batch
        seed (int): random seed for reproducibility

    Attributes:
        memory (deque): circular buffer storing experience tuples
        experience (namedtuple): data structure representing a single experience tuple

    Methods:
        add(state, action, reward, next_state, done): adds new experience to memory
        sample(): returns randomized batch of experiences as PyTorch tensors
        __len__(): returns current number of stored experiences
    """

    def __init__(self, 
                 buffer_size: int, 
                 batch_size: int, 
                 seed: int):
        """
        Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): state of a random function
        """
        random.seed(seed)

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            typename="Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(self, 
            state: list, 
            action: int, 
            reward: float, 
            next_state: list, 
            done: bool) -> None:
        """
        Add a new experience to memory.

        Args:
            state (list): current state observation vector
            action (int): action taken in current state
            reward (float): immediate reward received after action
            next_state (list): next state observation after action
            done (bool): flag, if there is no action left
        """
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self) -> Tuple[torch.Tensor]:
        """
        Randomly sample a batch of experiences from memory.

        Returns:
            (Tuple): contains
                - states (float32 tensor [batch_size, state_dim])
                - actions (int64 tensor [batch_size, 1])
                - rewards (float32 tensor [batch_size, 1])
                - next_states (float32 tensor [batch_size, state_dim])
                - dones (float32 tensor [batch_size, 1])
        """
        experiences = random.sample(
            population=self.memory, 
            k=min(self.batch_size, len(self.memory))
        )

        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8)).float().to(DEVICE)
        

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """
        Return the current size of internal memory.

        Returns:
            (int): current memory size
        """
        return len(self.memory)
