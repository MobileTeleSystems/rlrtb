import os
import torch
import random
import numpy as np

from abc import ABC, abstractmethod
from configparser import ConfigParser

from utils.enums import StateKeys, ConfigSection, ConfigParameters


class RLBaseBidAgent(ABC):
    """
    Base agent class for managing bids in ad auctions. Provides common functionality and interface for all agent 
    implementations.

    Defines:
    - core attributes for budget tracking and performance metrics
    - standardized methods for action selection and state updates
    - common budget exhaustion check
    - basic infrastructure for reinforcement learning agents

    Args:
        base_bid (int): fixed bid value from AD contract
        budget (int): remaining campaign budget
        seed (int): state of a random function

    Attributes:
        time_steps (int): number of time steps in an episode
        state_size (int): state space dimension
        batch_size (int): batch size for training
        rem_budget (int): remaining budget
        ctl_lambda (float): multiplier for adjusting bids
        total_bids (int): lifetime bid participation count
        total_wins (int): lifetime auction wins count
        total_rewards (float): cumulative rewards from clicks
        total_clicks (int): total clicks obtained
        total_budget_spend (float): total expenditure across auctions

    Methods:
        act: abstract method to select an action (implementation dependent)
        update: record auction outcomes and update budget
        done: check if the budget is exhausted
        _load_config: parses the config.cfg configuration file
        _reset_episode: reset the state when a new episode starts
        _set_seed: set all used seeds
        _reset_step: reset every time a new time step is entered
        _update_step: update statistics at each time step
        _update_reward_cost: update rewards and costs after taking action
        _get_state: generates the current agent state

    """
    def __init__(self, 
                 base_bid: int, 
                 budget: int, 
                 seed: int):
        self.base_bid = base_bid
        self.budget = budget
        self.seed = seed

        self._load_config()
        self._set_seed()
        self._reset_episode()

        self.total_bids = 0
        self.total_wins = 0
        self.total_clicks = 0
        self.total_rewards = 0.0
        self.total_budget_spend = 0

    def _load_config(self) -> None:
        """
        Parse the config.cfg file.
        """
        self.cfg = ConfigParser(allow_no_value=True)
        self.cfg.read(os.path.join(os.path.dirname(__file__), "config.cfg"))

        self.time_steps = int(self.cfg.get(ConfigSection.AGENT.value, ConfigParameters.TIME_STEPS.value, fallback=24))
        self.state_size = int(self.cfg.get(ConfigSection.AGENT.value, ConfigParameters.STATE_SIZE.value, fallback=7))
        self.batch_size = int(self.cfg.get(ConfigSection.AGENT.value, ConfigParameters.BATCH_SIZE.value, fallback=64))
        
    def _set_seed(self) -> None:
        """
        Set all seeds.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def _reset_episode(self) -> None:
        """
        Reset the state when episode changes. Update the budget and current statistics by resetting it to zero.

        rem_budget - remaining budget at time step t
        rol - the number of lambda regulation of opportunities left at t
        bcr - budget consumption rate:
            (self.budget - self.prev_budget) / self.prev_budget
        cpm - cost per mile of impressions between (t - 1) and t time steps:
            (self.prev_budget - self.running_budget) / self.cur_wins
        wr - winrate:
            wins_e / total_impressions
        """
        self.cur_time_step = 0
        self.rem_budget = self.budget
        self.rol = self.time_steps
        self.prev_budget = self.budget
        self.bcr = 0
        self.cpm = 0
        self.wr = 0

        self._reset_step()

        self.day = 1
        self.hour = 0
        self.ctl_lambda = 1.0
        self.wins_e = 0

    def _update_step(self) -> None:
        """
        Update the state with every bid request received for the state modeling.
        """
        self.cur_time_step += 1
        self.prev_budget = self.rem_budget
        self.rem_budget -= self.cost_t
        self.rol -= 1
        self.bcr = (self.rem_budget - self.prev_budget) / self.prev_budget if self.prev_budget != 0 else 0
        self.cpm = self.cost_t
        self.wr = self.wins_t / self.bids_t if self.bids_t else 0.0

    def _reset_step(self) -> None:
        """
        Reset every time a new time step is entered.
        """
        self.reward_t = 0.0
        self.cost_t = 0.0
        self.wins_t = 0 
        self.bids_t = 0

    def _update_reward_cost(self, 
                            reward: float, 
                            cost: float) -> None:
        """
        Update reward and cost statistics.
        """
        self.reward_t += reward
        self.cost_t += cost
        self.bids_t += 1

    def _get_state(self) -> np.ndarray:
        """
        Returns the state that will be used for the DQN state.

        Returns:
            (np.ndarray): algorithm state
        """
        return np.asarray(
            [
                self.cur_time_step,
                self.rem_budget,
                self.rol,
                self.bcr,
                self.cpm,
                self.wr,
                self.reward_t
            ]
        )

    @abstractmethod
    def act(self, 
            state: dict, 
            reward: float, 
            cost: float) -> float:
        """
        Taking action on every bid request (implementation depends on agent type).

        Args:
            state (dict): current state
            reward (dict): reward for completed action
            cost (float): bid price or market price according to auction type
        Returns:
            action (float): scaled bid price
        """
        pass

    def update(self, 
               observation: dict, 
               reward: float, 
               cost: float) -> None:
        """
        Updates the agent's internal state based on the outcome of the auction.

        Args:
            observation (dict): The observation for the current request.
            reward (float): The reward received for the bid.
            cost (float): The cost incurred for the bid.
        """
        if cost > 0:
            self.wins_t += 1
            self.wins_e += 1
            self.budget -= cost

            self.total_wins += 1
            self.total_rewards += reward
            self.total_budget_spend += cost

        self.total_clicks += observation.get(StateKeys.CLICK.value, 0) if reward else 0

    def done(self) -> bool:
        """
        Returns bool flag if there is no action left.

        Returns:
            bool: there is no actions left
        """
        return self.budget <= self.base_bid