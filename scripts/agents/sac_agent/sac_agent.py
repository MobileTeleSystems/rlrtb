import torch
import random
import numpy as np

from sac_agent.sac import SAC
from utils.replay_buffer import ReplayBuffer
from utils.rl_base_agent import RLBaseBidAgent
from utils.enums import ConfigParameters, ConfigSection, StateKeys


class SACBidAgent(RLBaseBidAgent):
    """
    Agent that makes bids using Soft Actor-Critic (SAC) reinforcement learning algorithm.

    Implements:
    - continuous action space for fine-grained bid adjustments
    - experience replay buffer for stable training
    - entropy-regulated exploration for balance between exploitation/exploration
    - adaptive bid scaling based on temporal and budgetary constraints

    Parameters:
        buffer_size (int): maximum capacity of experience storage
        batch_size (int): number of experiences per training batch
        seed (int): seed value for random number generator

    Attributes:
        state_size (int): dimension of state vector, loaded from config.cfg
        action_size: (int) dimension of action vector, loaded from config.cfg
        replay_buffer_size (int): maximum size of experience replay buffer, loaded from config.cfg
        batch_size (int): number of experiences per training batch, loaded from config.cfg
        sac_agent (SAC): SAC algorithm implementation
        memory (ReplayBuffer): experience replay buffer
        ctl_lambda (float): dynamic bid scaling factor
        day (StateKeys): current weekday, used to track episode boundaries
        hour (StateKeys): current hour of the day, used to track time step transitions
        pCTR (StateKeys): current pCTR of the day, used for bid scaling

    Methods:
        act: generate a scaled bid response
        _load_config: parses the config.cfg configuration file
        _get_state: return SAC state
        _reset_episode: reset the state when episode changes
    """

    def _load_config(self) -> None:
        """
        Parse algorithm-specific parameters from the config.cfg file.
        """
        super()._load_config()

        self.action_size = int(self.cfg.get(ConfigSection.AGENT.value, ConfigParameters.ACTION_SIZE.value, fallback=1))
        self.replay_buffer_size = int(self.cfg.get(ConfigSection.AGENT.value, ConfigParameters.REPLAY_BUFFER_SIZE.value, fallback=10000))

    def __init__(self, 
                 base_bid: int, 
                 budget: int, 
                 seed: int):
        """
        Initialize SAC based bidding Agent.

        Args:
            base_bid (int): bid price from contract
            budget (int): AD campaign's budget
            seed (int): state of a random function
        """
        super().__init__(base_bid, budget, seed)

        self._reset_episode()

        self.sac_agent = SAC(
            state_size=self.state_size, 
            action_size=self.action_size, 
            seed=seed
        )

        self.memory = ReplayBuffer(
            buffer_size=self.replay_buffer_size, 
            batch_size=self.batch_size, 
            seed=seed
        )

        self.sac_state = None
        self.sac_action = 0.0
        self.sac_reward = 0

    def _reset_episode(self) -> None:
        """
        Reset episode with SAC-specific components.
        """
        super()._reset_episode()

        self.sac_state = None
        self.sac_action = 0.0
        self.sac_reward = 0
        self.ctl_lambda = 1.0  # reset bid multiplier

    def act(self, 
            state: dict, 
            reward: float, 
            cost: float) -> float:
        """
        Taking action on every bid request.

        Args:
            state (dict): current state
            reward (dict): reward for completed action
            cost (float): bid price or market price according to auction type
        Returns:
            action (float): scaled bid price
        """
        episode_done = state.get(StateKeys.WEEKDAY.value) != self.day

        if state.get(StateKeys.HOUR.value) == self.hour and state.get(StateKeys.WEEKDAY.value) == self.day:
            self._update_reward_cost(
                reward=reward, 
                cost=cost
            )
        elif state.get(StateKeys.HOUR.value) != self.hour and state.get(StateKeys.WEEKDAY.value) == self.day:
            self._update_step()

            sac_next_state = self._get_state()

            if self.sac_state is not None:
                self.memory.add(
                    self.sac_state,
                    [self.sac_action],
                    self.sac_reward,
                    sac_next_state,
                    episode_done,
                )

            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.sac_agent.learn(experiences)

            action_scale = self.sac_agent.act(sac_next_state)  # [-1;1]
            self.sac_action = action_scale[0]  # extract the scalar action

            self.ctl_lambda *= (
                1 + self.sac_action * 0.1
            )  # scale the action to a reasonable range
            self.hour = state.get(StateKeys.HOUR.value)

            self.sac_state = sac_next_state
            self.sac_reward = reward

            self._reset_step()
            self._update_reward_cost(
                reward=reward, 
                cost=cost
            )

        elif state.get(StateKeys.WEEKDAY.value) != self.day:
            self._reset_episode()
            self.day = state.get(StateKeys.WEEKDAY.value)
            self.hour = state.get(StateKeys.HOUR.value)

            self._update_reward_cost(
                reward=reward, 
                cost=cost
            )

        self.total_bids += 1

        action = min(self.ctl_lambda * self.base_bid * state.get(StateKeys.PCTR.value, 0), self.base_bid)

        return action
