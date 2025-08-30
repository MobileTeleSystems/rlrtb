import torch
import numpy as np
import torch.nn.functional as F

from typing import Tuple
from torch.optim import Adam

from utils.actor import Actor
from utils.critic import Critic
from utils.rl_algorithm import RLAlgorithm
from utils.enums import ConfigSection, ConfigParameters

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3(RLAlgorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm implementation.

    TD3 addresses function approximation errors in DDPG through three key features:
    1. Twin Q-networks for clipped double Q-learning
    2. Delayed policy updates
    3. Target policy smoothing with clipped noise

    Parameters:
        state_size (int): Dimension of state space
        action_size (int): Dimension of action space
        seed (int): Random seed for reproducibility

    Attributes:
        actor_local (Actor): current policy network
        actor_target (Actor): target policy network
        critic_local_1 (Critic): first Q-network
        critic_target_1 (Critic): first target Q-network
        critic_local_2 (Critic): second Q-network
        critic_target_2 (Critic): second target Q-network
        batch_size (int): number of experiences per training batch, loaded from config.cfg
        update_every (int): frequency of network updates (in steps), loaded from config.cfg
        gamma (float): discount factor, loaded from config.cfg
        tau (float): soft update interpolation parameter, loaded from config.cfg
        policy_noise (float): noise scale for target smoothing, loaded from config.cfg
        noise_clip (float): noise clipping range, loaded from config.cfg
        policy_freq (int): policy update frequency, loaded from config.cfg
        actor_lr (float): actor learning rate, loaded from config.cfg
        critic_lr (float): critic learning rate, loaded from config.cfg

    Methods:
        act: generate action for given state using current policy
        learn: update networks using batch of experience tuples
        setup_networks: initialize actor and critic networks
        setup_optimizers: configure optimizers for actors and critics networks
        setup_memory: initialize experience replay buffer (unused in this implementation)
        _load_config: parses the config.cfg configuration file

    The agent loads configuration parameters from `config.cfg` file to initialize key attributes.
    """

    def _load_config(self) -> None:
        """
        Parse algorithm-specific parameters from the config.cfg file.
        """
        super()._load_config() 

        self.update_every = int(self.cfg.get(ConfigSection.TD3, ConfigParameters.UPDATE_EVERY.value, fallback=4))
        self.policy_noise = float(self.cfg.get(ConfigSection.TD3, ConfigParameters.POLICY_NOISE.value, fallback=0.2))
        self.noise_clip = float(self.cfg.get(ConfigSection.TD3, ConfigParameters.NOISE_CLIP.value, fallback=0.5))
        self.policy_freq = int(self.cfg.get(ConfigSection.TD3, ConfigParameters.POLICY_FREQ.value, fallback=2))

    def __init__(self, 
                 state_size: int, 
                 action_size: int, 
                 seed: int):
        """
        Initialize an TD3 agent object.

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): state of a random function
        """
        super().__init__(state_size, action_size, seed, ConfigSection.TD3)

        self.learn_step_counter = 0

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

        self.critic_local_1 = Critic(
            state_size=self.state_size, 
            action_size=self.action_size, 
            seed=self.seed
        ).to(DEVICE)

        self.critic_target_1 = Critic(
            state_size=self.state_size, 
            action_size=self.action_size, 
            seed=self.seed
        ).to(DEVICE)

        self.critic_local_2 = Critic(
            state_size=self.state_size, 
            action_size=self.action_size, 
            seed=self.seed
        ).to(DEVICE)

        self.critic_target_2 = Critic(
            state_size=self.state_size, 
            action_size=self.action_size, 
            seed=self.seed
        ).to(DEVICE)

        self.soft_update(
            local_model=self.actor_local, 
            target_model=self.actor_target
        )
        self.soft_update(
            local_model=self.critic_local_1, 
            target_model=self.critic_target_1
        )
        self.soft_update(
            local_model=self.critic_local_2, 
            target_model=self.critic_target_2
        )

    def setup_optimizers(self) -> None:
        """
        Configure optimizers for actors and critics networks.
        """
        self.actor_optimizer = Adam(
            params=self.actor_local.parameters(), 
            lr=self.actor_lr
        )

        self.critic_optimizer_1 = Adam(
            params=self.critic_local_1.parameters(), 
            lr=self.critic_lr
        )

        self.critic_optimizer_2 = Adam(
            params=self.critic_local_2.parameters(), 
            lr=self.critic_lr
        )

    def setup_memory(self) -> None:
        """
        Initialize experience replay buffer.
        """
        pass

    def act(self, 
            state: np.ndarray, 
            noise_scale: float = 0.1) -> torch.Tensor:
        """
        Returns an action for a given state as per current policy.

        Args:
            state (np.ndarray): current state
            noise_scale (float): scale of the noise added to the action

        Returns:
            (torch.Tensor): action taken in the current state
        """
        state = torch.FloatTensor(state).to(DEVICE)
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy().flatten()

        self.actor_local.train()
        noise = np.random.normal(
            loc=0, 
            scale=noise_scale, 
            size=self.action_size
        )

        return np.clip(action + noise, -1, 1)

    def learn(self, 
              experiences: Tuple[torch.Tensor]):
        """
        Update policy and value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
        """
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            # select action according to policy and add clipped noise
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)

            target_Q1 = self.critic_target_1(next_states, next_actions)
            target_Q2 = self.critic_target_2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q1 = self.critic_local_1(states, actions)
        current_Q2 = self.critic_local_2(states, actions)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        critic_loss.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.policy_freq == 0:
            actor_loss = -self.critic_local_1(states, self.actor_local(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(
                local_model=self.critic_local_1, 
                target_model=self.critic_target_1
            )
            self.soft_update(
                local_model=self.critic_local_2, 
                target_model=self.critic_target_2
            )
            self.soft_update(
                local_model=self.actor_local, 
                target_model=self.actor_target
            )
