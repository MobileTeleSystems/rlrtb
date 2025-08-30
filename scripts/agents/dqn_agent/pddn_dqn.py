import torch
import random
import numpy as np
import torch.nn.functional as F

from typing import Tuple
from torch.optim import Adam

from utils.model import QNetwork
from utils.replay_buffer import ReplayBuffer
from utils.model_dueling import DuelingQNetwork
from utils.rl_algorithm import RLAlgorithm
from utils.enums import ConfigSection, ConfigParameters, QNetworkType, ReplayBufferType, DQNType, LinearLayerType
from dqn_agent.per import PrioritizedReplayBuffer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PDDNDQN(RLAlgorithm):
    """
    Prioritized Deep Double/Dueling DQN (PDDNDQN) agent implementation with multiple extensions for improved learning
    stability and sample efficiency.

    Combines several DQN improvements in a modular architecture:
    - Double Q-learning (vanilla/double)
    - Dueling network architectures (regular/dueling)
    - Adaptive experience replay (regular/prioritized)
    - Advanced exploration strategies (epsilon-greedy/noisy networks)
    - Flexible linear layer configurations (regular/noised)

    The agent is configured through parameters in config.cfg.

    Parameters:
        state_size (int): dimension of state space
        action_size (int): dimension of action space
        seed (int): state of a random function

    Attributes:
        q_network_type (str/class): type of Q-network architecture (QNetwork/DuelingQNetwork), loaded from config.cfg
        replay_buffer_type (str): type of experience replay buffer (regular/prioritized), loaded from config.cfg
        linear_layer_type (str): type of neural network layers (regular/noised), loaded from config.cfg
        dqn_type (str): DQN variant (vanilla/double), loaded from config.cfg
        replay_buffer_size (int): maximum size of experience replay buffer, loaded from config.cfg
        batch_size (int): number of experiences per training batch, loaded from config.cfg
        gamma (float): discount factor for future rewards, loaded from config.cfg
        tau (float): soft update interpolation parameter, loaded from config.cfg
        lr (float): learning rate for Q-network optimization, loaded from config.cfg
        update_every (int): frequency of network updates (in steps), loaded from config.cfg
        qnetwork_local (torch.nn.Module): online Q-network
        qnetwork_target (torch.nn.Module): target Q-network
        optimizer (torch.optim): optimizer for Q-network training
        memory (ReplayBuffer): experience replay buffer
        t_step (int): counter for update timing

    Methods:
        step: stores experience and triggers learning
        act: selects action using ε-greedy or noisy network policy
        learn: updates Q-network parameters
        setup_networks: initialize Q-networks architectures
        setup_optimizers: configure optimizers for actors and critics networks
        setup_memory: initialize experience replay buffer
        _load_config: parses the config.cfg configuration file

    The class loads configuration parameters from `config.cfg` file to initialize key attributes.
    """

    def _load_config(self):
        """
        Parse algorithm-specific parameters from the config.cfg file.. Validates parameters and sets default values for 
        missing configuration entries.

        Loaded parameters include:
        - DQN architecture type (dueling/regular)
        - Replay buffer type (prioritized/regular)
        - Linear layer type (noised/regular)
        - DQN variant (double/vanilla)
        - Training hyperparameters (buffer size, batch size, gamma, tau, etc.)
        """
        super()._load_config()

        self.q_network_type = str(self.cfg.get(ConfigSection.DQN.value, ConfigParameters.Q_NETWORK_TYPE.value, fallback=QNetworkType.REGULAR.value))
        
        if self.q_network_type not in QNetworkType:
            print("The DQN model type is incorrect. \nPossible model types: regular or dueling")

        self.replay_buffer_type = str(self.cfg.get(ConfigSection.DQN.value, ConfigParameters.REPLAY_BUFFER_TYPE.value, fallback=ReplayBufferType.REGULAR.value))
        
        if self.replay_buffer_type not in ReplayBufferType:
            print("The replay buffer type is incorrect. \nPossible buffer types: regular or prioritized")

        self.linear_layer_type = str(self.cfg.get(ConfigSection.DQN.value, ConfigParameters.LINEAR_LAYER_TYPE.value, fallback=LinearLayerType.REGULAR.value))
        
        if self.linear_layer_type not in LinearLayerType:
            print("The linear layer type is incorrect. \nPossible layer types: regular or noised")

        self.dqn_type = str(self.cfg.get(ConfigSection.DQN.value, ConfigParameters.DQN_TYPE.value, fallback=DQNType.DOUBLE.value))
        
        if self.dqn_type not in DQNType:
            print("The DQN type is incorrect. \nPossible dqn types: vanilla or double")

        self.replay_buffer_size = int(self.cfg.get(ConfigSection.DQN.value, ConfigParameters.REPLAY_BUFFER_SIZE.value, fallback=10000))
        self.update_every = int(self.cfg.get(ConfigSection.DQN.value, ConfigParameters.UPDATE_EVERY.value, fallback=4))

    def __init__(self, 
                 state_size: int, 
                 action_size: int, 
                 seed: int):
        """
        Initialize DQN agent
        t_step - time step for updating every update_every steps.

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): state of a random function
        """
        super().__init__(state_size, action_size, seed, ConfigSection.DQN.value)

        self.t_step = 0

    def setup_networks(self) -> None:
        """
        Definie Q-networks network architectures.
        """
        if self.q_network_type == QNetworkType.DUELING.value:
            self.q_network_type = DuelingQNetwork
        elif self.q_network_type == QNetworkType.REGULAR.value:
            self.q_network_type = QNetwork

        self.qnetwork_local = self.q_network_type(
            state_size=self.state_size,
            action_size=self.action_size,
            seed=self.seed,
            layer_type=self.linear_layer_type,
        ).to(DEVICE)

        self.qnetwork_target = self.q_network_type(
            state_size=self.state_size,
            action_size=self.action_size,
            seed=self.seed,
            layer_type=self.linear_layer_type,
        ).to(DEVICE)
        
    def setup_optimizers(self) -> None:
        """
        Definie and configure optimizer.
        """
        self.optimizer = Adam(
            params=self.qnetwork_local.parameters(), 
            lr=self.lr
        )

    def setup_memory(self) -> None:
        """
        Memory configuration.
        """
        if self.replay_buffer_type == ReplayBufferType.PRIORITIZED.value:
            self.memory = PrioritizedReplayBuffer(
                buffer_size=self.replay_buffer_size,
                batch_size=self.batch_size,
                seed=self.seed,
            )
        elif self.replay_buffer_type == ReplayBufferType.REGULAR.value:
            self.memory = ReplayBuffer(
                buffer_size=self.replay_buffer_size,
                batch_size=self.batch_size,
                seed=self.seed,
            )

    def step(self, 
             state: list, 
             action: int, 
             reward: float, 
             next_state: list, 
             done: bool) -> None:
        """
        Stores experience in replay memory and performs network updates at configured intervals (update_every). Handles
        both regular and PER buffer sampling.

        Args:
            state (list): current state
            action (int):  action in current state
            reward (float): reward for action taken
            next_state (list): the state the agent goes to after an action
            done (bool): flag, if there is no action left
        """
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                if self.replay_buffer_type == ReplayBufferType.PRIORITIZED.value:
                    experiences = self.memory.sample_prioritized()
                elif self.replay_buffer_type == ReplayBufferType.REGULAR.value:
                    experiences = self.memory.sample()

                self.learn(experiences)

    def act(self, 
            state: list, 
            eps: float = 0.0) -> np.int64:
        """
        Returns actions for given state according to Epsilon-greedy policy if noisy layers not used.

        Implements:
        - Pure greedy action selection with noisy networks
        - ε-greedy exploration with regular networks

        Args:
            state (list): current state observation
            eps (float): exploration rate (ε) for ε-greedy selection
        Returns:
            (np.int64): selected action index
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.qnetwork_local.eval()

        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        self.qnetwork_local.train()

        if self.linear_layer_type == LinearLayerType.NOISED.value:
            return np.argmax(action_values.cpu().data.numpy())
        elif self.linear_layer_type == LinearLayerType.REGULAR.value:
            if random.random() > eps:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))

    def learn(self, 
              experiences: Tuple[torch.Tensor]) -> None:
        """
        Update value parameters using given batch of experience tuples:
            - get maximum predicted Q-values for the next states from target model
            - compute target Q-values for current states
            - get expected Q-values from local model
            - compute loss
            - minimize loss
            - update target network

        Args:
            experiences (Tuple[torch.Tensor]): batch of experiences from replay buffer (includes priorities/weights for
            PER)
        """
        if self.replay_buffer_type == ReplayBufferType.PRIORITIZED.value:
            states, actions, rewards, next_states, dones, indices, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        if self.dqn_type == DQNType.VANILLA.value:
            q_targets_next = self.qnetwork_target(next_states).max(dim=1, keepdim=True)[0]
            q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
        elif self.dqn_type == DQNType.DOUBLE.value:
            q_prime_actions = self.qnetwork_local(next_states).argmax(dim=1).unsqueeze(-1)
            q_prime_targets = self.qnetwork_target(next_states).gather(1, q_prime_actions)
            q_targets = rewards + self.gamma * q_prime_targets * (1 - dones)

        q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets).mean()

        if self.replay_buffer_type == ReplayBufferType.PRIORITIZED.value:
            priorities = loss.detach().cpu().data.numpy() * 1.0 * weights
            self.memory.update_priorities(indices, priorities)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.linear_layer_type == LinearLayerType.NOISED.value:
            self.qnetwork_local.layer_type.reset_noise
            self.qnetwork_target.layer_type.reset_noise

        self.soft_update(
            local_model=self.qnetwork_local,
            target_model=self.qnetwork_target,
        )
