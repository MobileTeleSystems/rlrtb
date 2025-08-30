import torch
import numpy as np
import torch.nn.functional as F

from typing import Tuple
from torch.optim import Adam

from utils.actor import Actor
from utils.critic import Critic
from utils.ounoise import OUNoise
from utils.replay_buffer import ReplayBuffer
from utils.rl_algorithm import RLAlgorithm
from utils.enums import ConfigSection, ConfigParameters

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG(RLAlgorithm):
    """
    Deep Deterministic Policy Gradient (DDPG) agent implementation.

    Off-policy actor-critic (AC) algorithm for continuous action spaces. Key components:
    - AC rchitecture with target networks
    - Experience replay buffer
    - Ornstein-Uhlenbeck noise for exploration
    - Soft target network updates

    Parameters:
        state_size (int): dimension of state space
        action_size (int): dimension of action space
        seed (int): state of a random function

    Attributes:
        actor_local (Actor): current policy network
        actor_target (Actor): target policy network
        critic_local (Critic): Q-value estimator network
        critic_target (Critic): target Q-value network
        memory (ReplayBuffer): experience replay buffer
        noise (OUNoise): exploration noise process
        gamma (float): discount factor for future rewards, loaded from config.cfg
        tau (float): soft update interpolation parameter, loaded from config.cfg
        actor_lr (float): actor learning rate, loaded from config.cfg
        critic_lr (float): critic learning rate, loaded from config.cfg
        weight_decay: L2 regularization for critic, loaded from config.cfg
        replay_buffer_size (int): maximum size of experience replay buffer, loaded from config.cfg
        batch_size (int): number of experiences per training batch, loaded from config.cfg
        update_every (int): frequency of network updates (in steps), loaded from config.cfg
        noise_theta: OU noise drift coefficient, loaded from config.cfg
        noise_sigma: OU noise volatility, loaded from config.cfg

    Methods:
        step: stores experience and triggers learning
        act: selects action using ε-greedy or noisy network policy
        learn: updates Q-network parameters
        reset_noise: reset exploration noise to initial state
        setup_networks: initialize actor and critic networks
        setup_optimizers: configure optimizers for actors and critics networks
        setup_memory: initialize experience replay buffer (unused in this implementation)
        _load_config: parses the config.cfg configuration file

    The class loads configuration parameters from `config.cfg` file to initialize key attributes.
    """

    def _load_config(self):
        """
        Parse algorithm-specific parameters from the config.cfg file.
        """
        super()._load_config()

        self.weight_decay = float(self.cfg.get(ConfigSection.DDPG.value, ConfigParameters.WEIGHT_DECAY.value, fallback=0))
        self.replay_buffer_size = int(self.cfg.get(ConfigSection.DDPG.value, ConfigParameters.REPLAY_BUFFER_SIZE.value, fallback=10000))
        self.update_every = int(self.cfg.get(ConfigSection.DDPG.value, ConfigParameters.UPDATE_EVERY.value, fallback=1))
        self.noise_theta = float(self.cfg.get(ConfigSection.DDPG.value, ConfigParameters.NOISE_THETA.value, fallback=0.15))
        self.noise_sigma = float(self.cfg.get(ConfigSection.DDPG.value, ConfigParameters.NOISE_SIGMA.value, fallback=0.2))

    def __init__(self, 
                 state_size: int, 
                 action_size: int, 
                 seed: int):
        """
        Initialize a DDPG agent object.

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(state_size, action_size, seed, ConfigSection.DDPG.value)

        self.noise = OUNoise(
            size=action_size, 
            seed=seed, 
            theta=self.noise_theta, 
            sigma=self.noise_sigma
        )
        self.t_step = 0

    def setup_networks(self) -> None:
        """
        Initialize Actor and Critic networks.
        """
        self.actor_local = Actor(
            state_size=self.state_size, 
            action_size=self.action_size, 
            seed=self.seed
        ).to(DEVICE)

        self.actor_target = Actor(
            state_size=self.state_size, 
            action_size=self.action_size, 
            seed=self.seed
        ).to(DEVICE)

        self.critic_local = Critic(
            state_size=self.state_size, 
            action_size=self.action_size, 
            seed=self.seed
        ).to(DEVICE)

        self.critic_target = Critic(
            state_size=self.state_size, 
            action_size=self.action_size, 
            seed=self.seed
        ).to(DEVICE)

    def setup_optimizers(self) -> None:
        """
        Configure optimizers for actors and critics networks.
        """
        self.actor_optimizer = Adam(
            params=self.actor_local.parameters(),
            lr=self.actor_lr
        )
        
        self.critic_optimizer = Adam(
            self.critic_local.parameters(),
            lr=self.critic_lr,
            weight_decay=self.weight_decay
        )

    def setup_memory(self) -> None:
        """
        Initialize experience replay buffer.
        """
        self.memory = ReplayBuffer(
            buffer_size=self.replay_buffer_size,
            batch_size=self.batch_size,
            seed=self.seed
        )

    def step(self, 
             state: list, 
             action: int, 
             reward: float, 
             next_state: list, 
             done: bool) -> None:
        """
        Stores experience in replay memory and performs network updates at configured intervals (update_every)

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
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, 
            state: np.ndarray, 
            add_noise: bool = True) -> np.ndarray:
        """
        Generate action for given state using current policy.

        Args:
            state (np.ndarray): current state
            add_noise (bool): whether to add exploration noise

        Returns:
            (np.ndarray): clipped action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def reset_noise(self) -> None:
        """
        Reset exploration noise process to initial state.
        """
        self.noise.reset()

    def learn(self, 
              experiences: Tuple[torch.Tensor]) -> None:
        """
        Update policy and value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
        """
        states, actions, rewards, next_states, dones = experiences

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  # gradient clipping
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(
            local_model=self.critic_local, 
            target_model=self.critic_target
        )
        self.soft_update(
            local_model=self.actor_local, 
            target_model=self.actor_target
        )

