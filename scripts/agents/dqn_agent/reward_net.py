import os
import torch
import random
import torch.nn.functional as F

from typing import Tuple
from torch.optim import Adam
from collections import defaultdict
from configparser import ConfigParser

from utils.model import QNetwork
from utils.enums import ConfigSection, ConfigParameters
from utils.reward_net_replay_buffer import RewardNetReplayBuffer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RewardNet:
    """
    A neural network model that predicts rewards for state-action pairs by interacting with and learning from the environment.
    The model uses experience replay and gradient descent to learn a reward function from observed transitions.

    Based on the paper "Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising"
    https://arxiv.org/abs/1802.08365

    Parameters:
        state_action_size (int): dimension of state-action pair
        reward_size (int): dimension of reward
        seed (int) - seed value for random number generators

    Attributes:
        reward_net (QNetwork): neural network for reward prediction (input: state-action, output: reward)
        memory (ReplayBuffer): experience replay buffer storing (state_action, reward) transitions
        optimizer (Adam): optimizer for training the reward network
        reward_dict (defaultdict): cache for quick reward lookup of seen state-action pairs
        state_action_size (int): dimension of the concatenated state-action vector
        reward_size (int): output dimension (typically 1 for scalar rewards)
        buffer_size (int): maximum capacity of experience storage, loaded from config.cfg
        batch_size (int): mini-batch size sampled from replay buffer during training, loaded from config.cfg
        lr (float): learning rate for the optimizer, loaded from config.cfg

    Methods:
        add: store a single experience in replay memory
        add_to_reward_dict: cache a reward for later fast retrieval
        get_from_reward_dict: retrieve cached rewards (returns 0 if missing)
        step: trigger learning when sufficient experiences are available
        act: get reward prediction for a given state-action pair
        learn: update network weights via backpropagation
        _load_config: parses the config.cfg configuration file

    The class loads configuration parameters from `config.cfg` file to initialize key attributes.
    """

    def _load_config(self) -> None:
        """
        Parse the config.cfg file.
        """
        cfg = ConfigParser(allow_no_value=True)
        cfg.read(os.path.join(os.path.dirname(__file__), "config.cfg"))

        self.buffer_size = int(cfg.get(ConfigSection.REWARD_NET.value, ConfigParameters.BUFFER_SIZE.value, fallback=10000))
        self.batch_size = int(cfg.get(ConfigSection.REWARD_NET.value, ConfigParameters.BATCH_SIZE.value, fallback=32))
        self.lr = float(cfg.get(ConfigSection.REWARD_NET.value, ConfigParameters.LR.value, fallback=1e-3))

    def __init__(self, 
                 state_action_size: int, 
                 reward_size: int, 
                 seed: int):
        """
        Initialize an RewardNet object.

        Args:
            state_action_size (int): dimension of state-action pair
            reward_size (int): dimension of reward
            seed (int): seed value for random number generators
        """
        self._load_config()
        
        random.seed(seed)

        self.state_action_size = state_action_size
        self.reward_size = reward_size

        self.reward_net = QNetwork(
            state_size=self.state_action_size, 
            action_size=self.reward_size, 
            seed=seed
        ).to(DEVICE)

        self.optimizer = Adam(
            params=self.reward_net.parameters(), 
            lr=self.lr
        )

        self.memory = RewardNetReplayBuffer(
            buffer_size=self.buffer_size, 
            batch_size=self.batch_size, 
            seed=seed
        )

        self.reward_dict = defaultdict()

    def add(self, 
            state_action: Tuple, 
            reward: float) -> None:
        """
        Save experience in replay memory.

        Args:
            state_action (Tuple): tuple with current state and selected action
            reward (float): reward for completed action
        """
        self.memory.add(state_action, reward)

    def add_to_reward_dict(self, 
                           sa: Tuple, 
                           reward: float) -> None:
        """
        Add state-action tuple to the reward dictonary.

        Args:
            sa (Tuple): tuple with current state and selected action
            reward (float): reward for completed action
        """
        self.reward_dict[sa] = reward

    def get_from_reward_dict(self, 
                             sa: Tuple) -> Tuple:
        """
        Get state-action tuple from the reward dictonary.

        Args:
            sa (Tuple): tuple with current state and selected action
        Returns:
            (Tuple): the value associated with the given state-action tuple in the reward dictionary, or 0 if not present.
        """
        return self.reward_dict.get(sa, 0)

    def step(self):
        """
        If enough samples are available in memory, get random subset and learn.
        """
        if len(self.memory) > 1:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, 
            state_action: Tuple) -> float:
        """
        Returns actions for given state as per current policy.

        Args:
            state_action (Tuple): tuple with current state and selected action
        Returns:
            (float): reward for completed action in current state
        """
        sa = torch.from_numpy(state_action).float().unsqueeze(0).to(DEVICE)

        return self.reward_net(sa)

    def learn(self, 
              experiences: Tuple[torch.Tensor]) -> None:
        """
        Update value parameters using given batch of experience tuples:
            - get expected reward values from RewardNet
            - compute loss
            - minimize loss

        Args:
            experiences (Tuple): tuple of (s, a, r, s', done) tuples
        """
        state_actions, rewards = experiences

        rnet_pred = self.reward_net(state_actions)
        loss = F.mse_loss(rnet_pred, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
