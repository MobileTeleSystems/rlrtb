import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
sys.path.append("../")

from typing import List, Tuple
from configparser import ConfigParser

from scripts.agents.utils.enums import StateKeys, SavingKeys, ConfigSection, AuctionTypes, ConfigParameters



class RTBEnv(gym.Env):
    """
    Real-Time Bidding (RTB) auction environment compatible with Gymnasium API, utilizing augmented iPinYou dataset for
    bid requests.
    https://gymnasium.farama.org/api/registry/

    Implements configurable RTB auction mechanics with:
    - Multiple auction types (first-price, second-price, VCG)
    - Customizable platform fees and auction steps
    - Noisy pCTR simulations
    - Multi-agent bidding support
    - Historical bid tracking and metrics

    The environment is configured through parameters in config.cfg.

    Parameters:
        num_agents (int): number of bidding agents in the environment

    Attributes:
        bid_requests (pd.DataFrame): loaded bid requests from dataset
        winning_bids_history (list): history of winning bids per auction
        winning_agents_history (list): history of winning agent IDs
        auction_steps_history (list): history of auction steps
        all_bids_history (list): complete history of all bids
        auction_type (str): auction mechanism type (first/second/vcg), loaded from config.cfg
        _data_path (str): path to the dataset, loaded from config.cfg
        _metric (str): optimization metric (clicks/TBD), loaded from config.cfg
        _step (int): current environment step counter
        _total_bids (int): total number of bids in dataset
        _total_steps (int): number of unique timestamps in dataset
        _seed (int): random seed for reproducibility, loaded from config.cfg
        _pctr_noise (float): noise level for pCTR values, loaded from config.cfg
        _platform_fee (float): platform fee percentage, loaded from config.cfg
        _auction_step (float): minimum bid increment step, loaded from config.cfg

    Methods:
        reset: initialize or reset environment state
        step: execute one timestep of the environment
        render: print statistics of the current state
        close: close and save environment
        _load_config: parse environment configuration from file
        _bid_state: update internal bid state tracking
        _get_observation: generate agent observation from bid request
        _get_rewards_costs: calculate rewards and costs for bids

    Note: the environment works correctly when the dataset consists of identical request blocks grouped by timestamp.
    For example, in this example of the dataset, blocks of 5 requests are used. The logic can be changed with little 
    effort, for example, by inserting fake requests without violating the general logic, but marking them with a flag 
    for agents.
    """

    def _load_config(self):
        """
        Load environment configuration from config.cfg file. Validates parameters and sets default values for missing
        configuration entries.

        Loaded parameters include:
        - Key metric (clicks/TBD)
        - Auction type (first/second/vcg)

        Raises AssertionError for invalid configuration values.
        """
        cfg = ConfigParser(allow_no_value=True)
        cfg.read(os.path.join(os.path.dirname(__file__), "config.cfg"))

        self._data_path = str(cfg.get(ConfigSection.DATA.value, ConfigParameters.DATA_PATH.value, fallback=""))

        if self._data_path == "":
            print("Be sure to specify the path to the data.")

        self._metric = str(cfg.get(ConfigSection.AUCTION.value, ConfigParameters.METRIC.value, fallback="clicks"))

        if self._metric not in ["clicks"]:
            print("The environment response metric is selected incorrectly. \nPossible values: clicks")

        self.auction_type = str(cfg.get(ConfigSection.AUCTION.value, ConfigParameters.AUCTION_TYPE.value, fallback=AuctionTypes.SECOND.value))

        if self.auction_type not in AuctionTypes:
            print("The auction type is selected incorrectly. \nPossible values: first, second, vcg")

        self._seed = int(cfg.get(ConfigSection.AUCTION.value, ConfigParameters.SEED.value, fallback=42))
        self._pctr_noise = float(cfg.get(ConfigSection.AUCTION.value, ConfigParameters.PCTR_NOISE.value, fallback=0.0))
        self._platform_fee = float(cfg.get(ConfigSection.AUCTION.value, ConfigParameters.PLATFORM_FEE.value, fallback=0.0))
        self._auction_step = float(cfg.get(ConfigSection.AUCTION.value, ConfigParameters.AUCTION_STEP.value, fallback=1.0))

    def __init__(self, 
                 num_agents: int):
        """
        Reading the config file and initializing the parameters.

        Args:
            num_agents (int): number of agents who place bids
        """
        super().__init__()

        self._load_config()
        np.random.seed(self._seed)

        self.num_agents = num_agents

        self.bid_requests = pd.read_parquet(
            path=self._data_path,
        )

        self._step = 0
        self._total_bids = len(self.bid_requests)
        self._total_steps = self.bid_requests[StateKeys.TIMESTAMP.value].nunique()

        self.winning_bids_history = []
        self.winning_agents_history = []
        self.auction_steps_history = []
        self.all_bids_history = []

    def reset(self, 
              seed: int = None, 
              options: str = None) -> Tuple[List[dict], float, float, bool]:
        """
        Reset the environment to an initial state.

        Args:
            seed (int): seed for random number generators to ensure reproducibility (currently unused)
            options (str): additional configuration options (currently unused)

        Returns:
            (Tuple): a tuple containing the following elements:
                observations (List[dict]): initial observations for each agent
                reward (float): initial environment configuration
                cost (float): initial environment configuration 
                done (bool): initial environment configuration
        """
        super().reset()
        self._block_index = 0

        self.bid_requests = self.bid_requests.sort_values(by=StateKeys.TIMESTAMP.value)
        self.blocks = [group for _, group in self.bid_requests.groupby(StateKeys.TIMESTAMP.value)]

        if len(self.blocks) == 0:
            raise ValueError("No data found in the provided CSV file.")

        bid_request_block = self.blocks[self._block_index]
        bid_request = bid_request_block.iloc[0]
        self.current_timestamp = bid_request[StateKeys.TIMESTAMP.value]

        self._bid_state(bid_request)

        return (
            [self._get_observation(row) for _, row in bid_request_block.iterrows()],
            0.0,
            0.0,
            False,
        )

    def step(self, 
             actions: List[List[float]]) -> Tuple[List[dict], List[List[float]], List[List[float]], bool]:
        """
        Take a step in the environment.

        Args:
            actions (List[List[float]]): actions taken by each agent
        Returns:
            (Tuple): a tuple containing the following elements:
                observations (List[dict]): observations for each agent after taking action
                rewards (List[List[float]]): rewards for each agent for each request
                cost (List[List[float]]): bid_price or market_price with auction_step according to auction_type for each
                request
                done (bool): flag if there is no action left
        """
        rewards = []
        costs = []
        observations = []

        if self._block_index >= len(self.blocks):
            return (
                [self._get_observation(pd.Series())] * self.num_agents,
                rewards,
                costs,
                True,
            )

        bid_request_block = self.blocks[self._block_index].reset_index(drop=True)
        self.current_timestamp = bid_request_block[StateKeys.TIMESTAMP.value].iloc[0]

        if len(actions) != len(bid_request_block):
            raise ValueError(
                f"Number of action lists ({len(actions)}) does not match the number of requests in the \
                             block ({len(bid_request_block)})."
            )

        current_block_bids = []

        self.all_bids_history.append(current_block_bids)

        for request_idx, row in bid_request_block.iterrows():
            self._bid_state(row)

            if request_idx < len(actions):
                agent_actions = actions[request_idx]

                if len(agent_actions) != self.num_agents:
                    raise ValueError(
                        f"Number of actions for request {request_idx} ({len(agent_actions)}) does not \
                                     match the number of agents ({self.num_agents})."
                    )

                req_rewards, req_costs = self._get_rewards_costs(agent_actions, row)
                self.all_bids_history.append(agent_actions)
            else:
                print(f"Warning: Not enough actions provided for request {request_idx}")
                req_rewards = [0.0] * self.num_agents
                req_costs = [0.0] * self.num_agents

            rewards.append(req_rewards)
            costs.append(req_costs)
            observations.append(self._get_observation(row))

        self._block_index += 1
        self._step += 1

        return observations, rewards, costs, False

    def _get_observation(self, 
                         row: pd.Series) -> dict:
        """
        Filling out the observation based on the received bid request.
        Click prob set to random. Must be replaced with model predict.

        Args:
            bid_request (pd.Series): received bid request
        Returns:
            observation (dict): filled observation
        """
        observation = {
            key: value for key, value in row.items() if key not in ["count", "index"]
        }

        return observation

    def _get_rewards_costs(self, 
                           actions: List[float], 
                           row: pd.Series) -> Tuple[List[float], List[float]]:
        """
        Compute reward according to given metric:
        - clicks, predicted click probability

        Implements three auction paradigms:
        1. First-price: Winner pays bid price
        2. Second-price: Winner pays second-highest bid + step
        3. VCG: Truthful mechanism calculating social welfare impact

        Args:
            action (List[float]): bid responses (bid_price)
            row (pd.Series): current bid request data
        Returns:
            (Tuple): a tuple containing the following elements:
                rewards (List[float]): reward from the environment for a completed action
                costs (List[float]): cost according to auction_type
        """
        rewards, costs = [0.0] * self.num_agents, [0.0] * self.num_agents

        if not actions:
            return rewards, costs

        highest_bid = np.max(actions)
        potential_winners = np.where(actions == highest_bid)[0]

        if len(potential_winners) > 1:
            winner_index = np.random.choice(potential_winners)
        else:
            winner_index = np.argmax(actions)

        winning_bid = actions[winner_index]
        reward, cost = 0, 0

        if self.auction_type == AuctionTypes.FIRST.value:
            if winning_bid >= row.get(StateKeys.SLOTPRICE.value):
                reward += 0.1

                if row.get(StateKeys.CLICK.value) == 1:
                    reward += 1

                cost = row.get(StateKeys.SLOTPRICE.value) + self._platform_fee

                self.winning_bids_history.append(cost)
                self.winning_agents_history.append(winner_index)

        elif self.auction_type == AuctionTypes.SECOND.value:
            if winning_bid >= row.get(StateKeys.PAYPRICE.value):
                reward += 0.1

                if row.get(StateKeys.CLICK.value) == 1:
                    reward += 1

                sorted_actions = sorted(actions, reverse=True)
                second_highest_bid = sorted_actions[1] if len(sorted_actions) > 1 else 0
                cost = second_highest_bid + self._auction_step + self._platform_fee

                self.winning_bids_history.append(winning_bid)
                self.winning_agents_history.append(winner_index)

        elif self.auction_type == AuctionTypes.VCG.value:
            if winning_bid >= row.get(StateKeys.SLOTPRICE.value):
                reward += 0.1

                if row.get(StateKeys.CLICK.value) == 1:
                    reward += 1

                social_welfare_with = 1 if row.get(StateKeys.CLICK.value) == 1 else 0

                actions_without_winner = actions[:winner_index] + actions[winner_index + 1 :]
                max_bid_without_winner = max(actions_without_winner) if actions_without_winner else 0
                social_welfare_without = 1 if max_bid_without_winner >= row.get(StateKeys.SLOTPRICE.value) and row.get(StateKeys.CLICK.value) == 1 else 0

                cost = social_welfare_without - social_welfare_with + self._platform_fee
                self.winning_bids_history.append(cost)
                self.winning_agents_history.append(winner_index)

        self.auction_steps_history.append(self._step)

        if (
            winning_bid >= row.get(StateKeys.SLOTPRICE.value)
            if self.auction_type != AuctionTypes.SECOND.value
            else winning_bid >= row.get(StateKeys.PAYPRICE.value)
        ):
            rewards[winner_index] = reward
            costs[winner_index] = cost

        return rewards, costs

    def _bid_state(self, 
                   row: pd.Series):
        """
        Update current state based on received bid request.

        Args:
            bid_request (pd.Series): received bid request
        """
        self.bid_price = row[StateKeys.BIDPRICE.value]
        self.slot_price = row[StateKeys.SLOTPRICE.value]
        self.pay_price = row[StateKeys.PAYPRICE.value]

        if self._pctr_noise:
            noisy_pCTR = np.clip(row[StateKeys.PCTR.value] + np.random.normal(0, 0.05), 0, 1)
            self.pCTR = noisy_pCTR
        else:
            self.pCTR = row[StateKeys.PCTR.value]

    def render_frame(self):
        """
        Display real-time auction state in human-readable format.
        """            
        if self._block_index >= len(self.blocks):
            print("\nAuction completed!")
            return
        
        if self._block_index != 0:
            print(f"\nCurrent timestamp: {self.current_timestamp}")
            print(f"Current step: {self._block_index}")

            if self.winning_bids_history and self.winning_agents_history and len(self.winning_bids_history) > self._step:
                winner_index = self.winning_agents_history[self._step]
                winning_bid = self.winning_bids_history[self._step]
                print(f"Current step winner: Agent {winner_index}, Bid: {winning_bid}")
            else:
                print("The winner of the current step has not yet been determined.")

    def close(self, 
              saving: bool = False):
        """
        Cleaning the environment and saving the data.

        Args:
            saving (bool) - do we want to save
        """
        if saving:
            if self.winning_bids_history:
                df_winning_bids = pd.DataFrame(
                    {
                        SavingKeys.STEP.value: range(len(self.winning_bids_history)),
                        SavingKeys.WINNING_BID.value: self.winning_bids_history,
                        SavingKeys.WINNING_AGENT.value: self.winning_agents_history,
                    }
                )
                df_winning_bids.to_csv("winning_bids_history.csv", index=False)

            if self.all_bids_history:
                all_bids_flat = []
                step = 0

                for step_bids in self.all_bids_history:
                    for request_bids in step_bids:
                        for agent_index, bid in enumerate(request_bids):
                            all_bids_flat.append(
                                {
                                    SavingKeys.STEP.value: step, 
                                    SavingKeys.AGENT_INDEX.value: agent_index, 
                                    SavingKeys.BID.value: bid
                                }
                            )
                            
                        step += 1

                df_all_bids = pd.DataFrame(all_bids_flat)
                df_all_bids.to_csv("all_bids_history.csv", index=False)

        if self.winning_bids_history:
            print("\nSummary:")
            print(f"Total steps: {self._block_index}")
