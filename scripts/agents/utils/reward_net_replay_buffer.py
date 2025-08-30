import torch
import random
import numpy as np

from typing import Tuple
from collections import namedtuple, deque

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RewardNetReplayBuffer:
    """
    The ReplayBuffer class implements a fixed-size experience replay buffer for Reward Net.

    This buffer stores experience tuples (state_action, reward) and provides functionality to randomly sample batches
    of experiences for training. Experiences are automatically evicted when the buffer reaches its maximum capacity to
    maintain temporal diversity in training samples.

    Parameters:
        buffer_size (int): maximum number of experience tuples the buffer can hold
        batch_size (int): number of experiences to sample per training batch
        seed (int): random seed for reproducibility

    Attributes:
        memory (deque): circular buffer storing experience tuples
        experience (namedtuple): data structure representing a single experience tuple

    Methods:
        add: adds new experience to memory
        sample: returns randomized batch of experiences as PyTorch tensors
        __len__: returns current number of stored experiences
    """

    def __init__(self, 
                 buffer_size: int, 
                 batch_size: int, 
                 seed: int):
        """
        Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): maximum size of buffer
            batch_size  (int): size of each training batch
            seed (int): state of a random function
        """
        random.seed(seed)

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            typename="Experience", 
            field_names=["state_action", "reward"]
        )

    def add(self, 
            state_action: Tuple, 
            reward: float) -> None:
        """
        Add a new experience to memory.

        Args:
            state_action (Tuple): tuple with current state and selected action
            reward (float): reward for action taken
        """
        experience = self.experience(state_action, reward)
        self.memory.append(experience)

    def sample(self) -> Tuple[Tuple, float]:
        """
        Randomly sample a batch of experiences from memory.

        Returns:
            (Tuple): batch of experiences
        """
        experiences = random.sample(
            population=self.memory, 
            k=min(self.batch_size, len(self.memory))
        )

        state_actions = torch.from_numpy(np.vstack([exp.state_action for exp in experiences if exp is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(DEVICE)

        return (state_actions, rewards)

    def __len__(self) -> int:
        """
        Return the current size of internal memory.

        Returns:
            (int): current memory size
        """
        return len(self.memory)
