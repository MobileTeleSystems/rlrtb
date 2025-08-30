import numpy as np

from ddpg_agent.ddpg import DDPG
from utils.rl_base_agent import RLBaseBidAgent
from utils.enums import ConfigParameters, ConfigSection, StateKeys


BETA = [-0.075, -0.050, -0.025, -0.0125, 0, 0.0125, 0.025, 0.050, 0.075]


class DDPGBidAgent(RLBaseBidAgent):
    """
    Agent that makes bids using Deep Deterministic Policy Gradient (DDPG) reinforcement learning algorithm (RL).

    Implements:
    - adaptive bid scaling based on predicted CTR (pCTR)
    - adaptive budget control using beta-scaled bids
    - episodic training with daily budget cycles
    - continuous action space with discrete beta selection

    Parameters:
        base_bid (int): fixed bid value from AD contract
        budget (int): remaining campaign budget
        seed (int): seed for random number generators to ensure reproducibility

     Attributes:
        ddpg_agent (DDPG): DDPG agent for policy learning
        ctl_lambda (float): dynamic bid scaling factor
        state_size (int): dimension of state vector, loaded from config.cfg
        eps_start (float): initial exploration rate, loaded from config.cfg
        eps_end (float): minimum exploration rate, loaded from config.cfg
        anneal (float): exploration rate decay factor, loaded from config.cfg
        action_size (int): size of the action space (continuous, i.e 1), representing a multiplier for the base bid, loaded from config.cfg
        action_scale (float): caling factor for DDPG actions, loaded from config.cfg
        action_offset (int): offset for beta selection mapping, loaded from config.cfg

    Methods:
        act: determines the bid price based on the current state and updates internal states
        _load_config: parses the config.cfg configuration file
        _reset_episode: reset the state and noise when episode changes
        _reset_step: add exploration rate decay to step reset

    The agent loads configuration parameters from `config.cfg` file to initialize key attributes.
    """

    def _load_config(self):
        """
        Parse algorithm-specific parameters from the config.cfg file.
        """
        super()._load_config()

        self.action_size = int(self.cfg.get(ConfigSection.AGENT.value, ConfigParameters.ACTION_SIZE.value, fallback=1))
        self.eps_start = float(self.cfg.get(ConfigSection.AGENT.value, ConfigParameters.EPS_START.value, fallback=0.95))
        self.eps_end = float(self.cfg.get(ConfigSection.AGENT.value, ConfigParameters.EPS_END.value, fallback=0.05))
        self.anneal = float(self.cfg.get(ConfigSection.AGENT.value, ConfigParameters.ANNEAL.value, fallback=1e-3))
        self.action_scale = float(self.cfg.get(ConfigSection.AGENT.value, ConfigParameters.ACTION_SCALE.value, fallback=1.0))
        self.action_offset = int(self.cfg.get(ConfigSection.AGENT.value, ConfigParameters.ACTION_OFFSET.value, fallback=4))

    def __init__(self,
                 base_bid: int,
                 budget: int,
                 seed: int):
        """
        Initialize DDPG agent with network and exploration parameters.

        Args:
            base_bid (int): bid price from contract
            budget (int): AD campaing's budget
            seed (int): state of a random function
        """
        super().__init__(base_bid, budget, seed)

        self.ddpg_agent = DDPG(
            state_size=self.state_size,
            action_size=self.action_size,  # сontinuous action space
            seed=self.seed
        )

        self.ddpg_agent.reset_noise() # direct noise reset after initialization
        
        self.ddpg_state = None
        self.ddpg_action = 0.0
        self.reward_dict = {}
        self.states = []
        self.values = 0

    def _reset_episode(self) -> None:
        """
        Reset the state and noise when episode changes.
        """
        super()._reset_episode()

        self.eps = self.eps_start
        self.values = 0

        if hasattr(self, 'ddpg_agent') and self.ddpg_agent is not None:
            self.ddpg_agent.reset_noise()

    def _reset_step(self) -> None:
        """
        Add exploration rate decay to step reset.
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

            ddpg_next_state = self._get_state()

            # get the action and map it to discrete if needed (DDPG uses continuous action space)
            action_beta_continuous = self.ddpg_agent.act(
                state=ddpg_next_state
            )
            action_beta_index = int(np.clip(np.round(action_beta_continuous * self.action_scale + self.action_offset), 0, len(BETA) - 1))

            self.ddpg_agent.step(
                state=self.ddpg_state, 
                action=self.ddpg_action, 
                reward=reward, 
                next_state=ddpg_next_state, 
                done=episode_done
            )
            self.ddpg_state = ddpg_next_state
            self.ddpg_action = action_beta_continuous

            self.ctl_lambda *= 1 + BETA[action_beta_index]
            self.hour = state.get(StateKeys.HOUR.value)

            self._reset_step()

            self._update_reward_cost(
                reward=reward, 
                cost=cost
            )

            self.values += self.reward_t
            self.states.append((self.ddpg_state, self.ddpg_action))

        elif state.get(StateKeys.WEEKDAY.value) != self.day:
            self.ddpg_agent.reset_noise()
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

