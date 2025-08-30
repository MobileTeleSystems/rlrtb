import random
import numpy as np

from dqn_agent.pddn_dqn import PDDNDQN
from dqn_agent.reward_net import RewardNet
from utils.rl_base_agent import RLBaseBidAgent
from utils.enums import ConfigParameters, ConfigSection, StateKeys

BETA = [-0.2, -0.1, -0.05, -0.025, 0, 0.025, 0.05, 0.1, 0.2]


class DQNBidAgent(RLBaseBidAgent):
    """
    Agent that makes a bid equal to the contract price, scaled by expected click-through rate (pCTR) using modified DQN.

    Implements extension of the basic DQN by:
    - a Double Dueling Q-networks for action selection (optional)
    - Prioritized Experience Replay as replay buffer (optinal)
    - a Reward Network for learning optimal reward signals
    - actions correspond to bid adjustment factors from BETA values
    - adaptive budget control using beta-scaled bids
    - episodic training with daily budget cycles
    - epsilon-greedy exploration with annealing

    Parameters:
        base_bid (int): fixed bid value from AD contract
        budget (int): remaining campaign budget
        seed (int): seed for random number generators to ensure reproducibility

    Attributes:
        dqn_agent (PDDNDQN): DQN for policy learning
        reward_net (RewardNet): network for reward signal approximation
        state_size (int): dimension of state vector, loaded from config.cfg
        eps_start (float): initial exploration rate, loaded from config.cfg
        eps_end (float): minimum exploration rate, loaded from config.cfg
        anneal (float): exploration rate decay factor, loaded from config.cfg
        action_size (int): size of the action space

    Methods:
        act: generate a scaled bid response
        _load_config: parses the config.cfg configuration file
        _reset_episode: reset state when episode changes
        _reset_step: add epsilon decay to step reset

    The agent loads configuration parameters from `config.cfg` file to initialize key attributes.
    """

    def _load_config(self) -> None:
        """
        Parse algorithm-specific parameters from the config.cfg file.
        """
        super()._load_config()

        self.eps_start = float(self.cfg.get(ConfigSection.AGENT.value, ConfigParameters.EPS_START.value, fallback=0.95))
        self.eps_end = float(self.cfg.get(ConfigSection.AGENT.value, ConfigParameters.EPS_END.value, fallback=0.05))
        self.anneal = float(self.cfg.get(ConfigSection.AGENT.value, ConfigParameters.ANNEAL.value, fallback=1e-3))

    def __init__(self, 
                 base_bid: int, 
                 budget: int, 
                 seed: int):
        """
        Initialize DQN agent with networks and exploration parameters.

        Args:
            base_bid (int): bid price from contract
            budget (int): AD campaing's budget
            seed (int): state of a random function
        """
        super().__init__(base_bid, budget, seed)

        self.action_size = len(BETA)

        self._reset_episode()

        self.dqn_agent = PDDNDQN(
            state_size=self.state_size, 
            action_size=self.action_size, 
            seed=seed
        )

        self.reward_net = RewardNet(
            state_action_size=self.state_size + 1, 
            reward_size=1, 
            seed=seed
        )

        self.dqn_state = self._get_state()
        self.dqn_action = random.sample(
            population=BETA, 
            k=1
        )[0]

        self.dqn_reward = 0
        self.reward_dict = {}
        self.states = []
        self.values = 0

    def _reset_episode(self) -> None:
        """
        Reset episode with DQN-specific components.
        """
        super()._reset_episode()

        self.eps = self.eps_start
        self.dqn_state = self._get_state()
        self.dqn_action = random.sample(
            population=BETA, 
            k=1
        )[0]
        self.states = []
        self.values = 0

    def _reset_step(self) -> None:
        """
        Add epsilon decay to step reset.
        """
        super()._reset_step()

        self.eps = max(self.eps_start - self.anneal * self.cur_time_step, self.eps_end)

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

            self.reward_net.step()
            dqn_next_state = self._get_state()

            action_beta = self.dqn_agent.act(state=dqn_next_state, eps=self.eps)

            sa = np.append(self.dqn_state, self.dqn_action)

            rnet_reward = float(self.reward_net.act(sa))

            self.dqn_agent.step(
                self.dqn_state,
                self.dqn_action,
                rnet_reward,
                dqn_next_state,
                episode_done,
            )
            self.dqn_state = dqn_next_state
            self.dqn_action = action_beta

            self.ctl_lambda *= 1 + BETA[action_beta]
            self.hour = state.get(StateKeys.HOUR.value)

            self._reset_step()

            self._update_reward_cost(
                reward=reward, 
                cost=cost
            )

            self.values += self.reward_t
            self.states.append((self.dqn_state, self.dqn_action))

        elif state.get(StateKeys.WEEKDAY.value) != self.day:
            for s, a in self.states:
                sa = tuple(np.append(s, a))
                max_r = max(self.reward_net.get_from_reward_dict(sa=sa), self.values)
                self.reward_net.add_to_reward_dict(sa=sa, reward=max_r)
                self.reward_net.add(state_action=sa, reward=max_r)

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

