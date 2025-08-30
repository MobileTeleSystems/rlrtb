import torch
import numpy as np
import torch.nn.functional as F

from typing import Tuple
from torch.optim import Adam

from utils.gaussian_policy import GaussianPolicy
from utils.double_q_network import DoubleQNetwork
from utils.rl_algorithm import RLAlgorithm
from utils.enums import ConfigSection, ConfigParameters

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SAC(RLAlgorithm):
    """
    Soft Actor-Critic (SAC) is an off-policy actor-critic deep RL algorithm that maximizes the expected return while
    also maximizing entropy. This implementation includes automatic entropy tuning.

    Parameters:
        state_size (int): dimension of state space
        action_size (int): dimension of action space
        seed (int): state of a random function

    Attributes:
        actor (GaussianPolicy): Gaussian policy network (actor)
        critic (DoubleQNetwork): Twin Q-network (critic)
        critic_target (DoubleQNetwork): target Q-network for stable learning
        actor_optimizer (Adam): optimizer for actor network
        critic_optimizer (Adam): optimizer for critic networks
        target_entropy (float): target entropy for automatic alpha tuning
        log_alpha (Tensor): learnable parameter for entropy temperature
        alpha_optim (Adam): optimizer for alpha parameter
        alpha (float): Entropy temperature coefficient, loaded from config.cfg
        gamma (float): discount factor, loaded from config.cfg
        tau (float): soft update interpolation parameter, loaded from config.cfg
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

        self.alpha = float(self.cfg.get(ConfigSection.SAC, ConfigParameters.ALPHA.value, fallback=32))

    def __init__(self, 
                 state_size: int, 
                 action_size: int, 
                 seed: int):
        """
        Initializes SAC algorithm.

        Args:
            state_size (int): dimension of state space
            action_size (int): dimension of action space
            seed (int): state of a random function
        """
        super().__init__(state_size, action_size, seed, ConfigSection.SAC)
        
        self.target_entropy = -torch.prod(torch.Tensor((action_size,)).to(DEVICE)).item()

    def setup_networks(self) -> None:
        """
        Initialize Actor and Critic networks.
        """
        self.actor = GaussianPolicy(
            state_size=self.state_size, 
            action_size=self.action_size, 
            seed=self.seed
        ).to(DEVICE)

        self.critic = DoubleQNetwork(
            state_size=self.state_size,
            action_size=self.action_size, 
            seed=self.seed
        ).to(DEVICE)

        self.critic_target = DoubleQNetwork(
            state_size=self.state_size, 
            action_size=self.action_size, 
            seed=self.seed
        ).to(DEVICE)

        self.critic_target.load_state_dict(self.critic.state_dict())

    def setup_optimizers(self) -> None:
        """
        Configure optimizers for actors and critics networks.
        """
        self.actor_optimizer = Adam(
            params=self.actor.parameters(), 
            lr=self.actor_lr
        )

        self.critic_optimizer = Adam(
            params=self.critic.parameters(), 
            lr=self.critic_lr
        )

        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)

        self.alpha_optim = Adam(
            params=[self.log_alpha], 
            lr=self.actor_lr
        )

    def setup_memory(self) -> None:
        """
        Initialize experience replay buffer.
        """
        pass

    def act(self, 
            state: np.ndarray) -> np.ndarray:
        """
        Returns actions for given state as per current policy.

        Args:
            state (np.ndarray): current state
        Returns:
            (np.ndarray): list of actions for given state
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        self.actor.eval()

        with torch.no_grad():
            action, _ = self.actor.sample(state)

        self.actor.train()

        return action.cpu().data.numpy().flatten()

    def learn(self, 
              experiences: Tuple[torch.Tensor]):
        """
        Update policy and value parameters using given batch of experience tuples:
        
        1. critic networks by minimizing Q-value MSE
        2. actor policy using policy gradient
        3. entropy temperature (alpha) using gradient descent
        4. target networks via soft update

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
        """
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            next_state_actions, next_state_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_state_actions)
            min_q_target = (
                torch.min(q1_target, q2_target) - self.alpha * next_state_log_probs
            )
            next_q_values = rewards + (1 - dones) * self.gamma * min_q_target

        q1_expected, q2_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(q1_expected, next_q_values) + F.mse_loss(q2_expected, next_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, pi)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        self.soft_update(
            local_model=self.critic_target, 
            target_model=self.critic
        )
