import os
import torch
import numpy as np

from typing import Tuple
from collections import deque
from configparser import ConfigParser

from utils.replay_buffer import ReplayBuffer
from utils.enums import ConfigSection, ConfigParameters

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Implementation of prioritized experience replay (PER) buffer based on the ReplayBuffer class. Prioritizes
    experiences based on temporal-difference (TD) error for more efficient learning.

    Implements extension of the basic experience replay by:
    1. Storing priority values for each experience
    2. Sampling experiences proportionally to their priority
    3. Using importance sampling weights to correct bias

    Based on the PER algorithm from "Prioritized Experience Replay" (Schaul et al., 2015)
    https://arxiv.org/abs/1511.05952

    Parameters:
        buffer_size (int): maximum capacity of experience storage
        batch_size (int): number of experiences per training batch
        seed (int): seed value for random number generator

    Attributes:
        alpha (float): priority exponent (0-1) controlling prioritization strength, loaded from config.cfg
        beta (float): importance sampling exponent (0-1) for bias correction, loaded from config.cfg
        eps (float): small constant to prevent zero probabilities, loaded from config.cfg
        priorities (deque): priority values for each stored experience
        max_priority (float): maximum priority value for new experiences
        memory (deque): inherited from ReplayBuffer - stores experience tuples
        experience (namedtuple): inherited data structure for experience storage

    Methods:
        add: stores experience with max priority
        update_priorities: updates priorities using TD-errors
        sample_prioritized: samples batch with priority-based selection
        _load_config: parses the config.cfg configuration file

    The class loads configuration parameters from `config.cfg` file to initialize key attributes.
    """

    def _load_config(self) -> None:
        """
        Parse the config.cfg file.
        """
        self.cfg = ConfigParser(allow_no_value=True)
        self.cfg.read(os.path.join(os.path.dirname(__file__), "config.cfg"))

        self.alpha = float(self.cfg.get(ConfigSection.PER.value, ConfigParameters.ALPHA.value, fallback=0.6))
        self.beta = float(self.cfg.get(ConfigSection.PER.value, ConfigParameters.BETA.value, fallback=0.4))
        self.eps = float(self.cfg.get(ConfigSection.PER.value, ConfigParameters.EPS.value, fallback=1e-3))

    def __init__(self, 
                 buffer_size: int, 
                 batch_size: int, 
                 seed: int):
        """
        Initialize PER instance.

        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): state of a random function
        """
        super().__init__(buffer_size, batch_size, seed)
        
        self._load_config()
        np.random.seed(seed)

        self.priorities = deque(maxlen=buffer_size)
        self.max_priority = 1.0

    def add(self, 
            state: list, 
            action: int, 
            reward: float, 
            next_state: list, 
            done: bool) -> None:
        """
        Store a new experience in the buffer with maximum priority.

        New experiences are added with the highest priority to ensure they are sampled at least once during initial
        training phases.

        Args:
            state (list): current state observation vector
            action (int): action taken in current state
            reward (float): immediate reward received after action
            next_state (list): next state observation after action
            done (bool): flag, if there is no action left
        """
        experience = self.experience(state, action, reward, next_state, done)
        
        self.memory.append(experience)
        self.priorities.append(self.max_priority)

    def update_priorities(self, 
                          indices: list, 
                          td_errors: list) -> None:
        """
        Update priorities array with new TD errors.

        Args:
            indices (list): priority indices
            td_errors (list): temporal-difference errors
        """
        td_errors_abs_eps = (td_errors.abs() + self.eps).cpu().detach().numpy().flatten()
        current_max = td_errors_abs_eps.max()
        
        self.max_priority = max(self.max_priority, current_max)

        for idx, td_error in zip(indices, td_errors_abs_eps):
            self.priorities[idx] = td_error.item()

    def sample_prioritized(self) -> Tuple[torch.Tensor]:
        """
        Sample a batch of experiences with priority-based selection and compute
        importance sampling weights.

        Implements:
        1. Priority -> Probability conversion using power law (alpha)
        2. Stochastic index selection based on probabilities
        3. Importance sampling weights calculation (beta)
        4. Experience batch preparation as PyTorch tensors

        Returns:
            (Tuple[torch.Tensor]): contains
                - states (float32 tensor [batch_size, state_dim])
                - actions (int64 tensor [batch_size, 1])
                - rewards (float32 tensor [batch_size, 1])
                - next_states (float32 tensor [batch_size, state_dim])
                - dones (float32 tensor [batch_size, 1])
                - indices (list[int]): buffer indices of sampled experiences
                - weights (float32 tensor [batch_size, 1]): importance weights
        """
        td_errors = torch.from_numpy(np.asarray(self.priorities, dtype=np.float64).flatten())
        probs = (np.abs(td_errors) + self.eps) ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(
            a=len(self.memory), 
            size=self.batch_size, 
            p=probs
        )

        states = torch.from_numpy(np.vstack([self.memory[idx].state for idx in indices])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([self.memory[idx].action for idx in indices])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([self.memory[idx].reward for idx in indices])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([self.memory[idx].next_state for idx in indices])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([self.memory[idx].done for idx in indices]).astype(np.uint8)).float().to(DEVICE)

        weights = (1.0 / (len(td_errors) * probs[indices])) ** self.beta
        weights = (weights / weights.max()).reshape(weights.shape[0], 1)

        return (states, actions, rewards, next_states, dones, indices, weights)
