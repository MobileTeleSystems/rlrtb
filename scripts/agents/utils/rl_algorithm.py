import os
import torch

from typing import Tuple
from abc import ABC, abstractmethod
from configparser import ConfigParser

from utils.enums import ConfigSection, ConfigParameters


class RLAlgorithm(ABC):
    """
    Base class for Reinforcement Learning algorithms providing common infrastructure and standardized interface.
    
    This abstract class implements shared functionality for RL agents and defines the template 
    for algorithm-specific implementations. It handles configuration loading, network/optimizer setup,
    and provides common utilities while enforcing implementation of key RL components.
    
    Parameters:
        state_size (int): dimensionality of the environment's state space
        action_size (int): dimensionality of the environment's action space
        seed (int): random seed for reproducibility
        config_section (ConfigSection): section name in config.cfg containing algorithm-specific parameters
        
    Attributes:
        cfg (ConfigParser): parsed configuration parameters
        gamma (float): discount factor for future rewards
        tau (float): soft update parameter for target networks
        actor_lr (float): learning rate for actor/policy networks
        critic_lr (float): learning rate for critic/value networks
        
    Methods:
        setup_networks: abstract method for defining neural network architectures
        setup_optimizers: abstract method for configuring optimizers
        setup_memory: abstract memory configuration method
        soft_update: performs Polyak averaging for target network updates
        act: abstract method for action selection
        learn: abstract method for training step implementation
        _load_config: parses configuration file and sets common hyperparameters

    Note: fallback parameter values if not in config for all methods are different.
    """
    def __init__(self, 
                 state_size: int, 
                 action_size: int, 
                 seed: int,
                 config_section: ConfigSection):
        """
        Initialize an Agent object.

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): state of a random function
            config_section (ConfigSection): section name in config.cfg
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.config_section = config_section

        self._load_config()
        self.setup_networks()
        self.setup_optimizers()
        self.setup_memory()
        
    def _load_config(self) -> None:
        """
        Parse the config.cfg file.
        """
        self.cfg = ConfigParser(allow_no_value=True)
        self.cfg.read(os.path.join(os.path.dirname(__file__), "config.cfg"))

        self.gamma = float(self.cfg.get(self.config_section, ConfigParameters.GAMMA.value, fallback=0.99))
        self.tau = float(self.cfg.get(self.config_section, ConfigParameters.TAU.value, fallback=0.001))
        self.batch_size = int(self.cfg.get(self.config_section, ConfigParameters.BATCH_SIZE.value, fallback=64))
        self.lr = float(self.cfg.get(self.config_section, ConfigParameters.LR.value, fallback=3e-3))
        self.actor_lr = float(self.cfg.get(self.config_section, ConfigParameters.ACTOR_LR.value, fallback=1e-3))
        self.critic_lr = float(self.cfg.get(self.config_section, ConfigParameters.CRITIC_LR.value, fallback=1e-3))
        
    @abstractmethod
    def setup_networks(self):
        """
        Define neural network architectures.
        """
        pass
        
    @abstractmethod
    def setup_optimizers(self) -> None:
        """
        Define and configure optimizers.
        """
        pass

    @abstractmethod
    def setup_memory(self) -> None:
        """
        Memory configuration.
        """
        pass

    @abstractmethod
    def act(self, 
            state: list,
            *args,
            **kwargs) -> None:
        """
        Setting the logic for taking actions.
        """
        pass
    
    @abstractmethod
    def learn(self, 
              experiences: Tuple[torch.Tensor]) -> None:
        """
        Perform an update using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): batch of experiences from replay buffer (includes priorities/weights for
            PER)
        """
        pass

    def soft_update(self, 
                    local_model: torch.nn.Module, 
                    target_model: torch.nn.Module) -> None:
        """
        Soft update model parameters using Polyak averaging.
        θ_target = τ * θ_local + (1 - τ) * θ_target

        Args:
            local_model (torch.nn.Module): weights will be copied from
            target_model (torch.nn.Module): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            