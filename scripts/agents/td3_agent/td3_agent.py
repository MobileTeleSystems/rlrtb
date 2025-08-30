import numpy as np

from td3_agent.td3 import TD3
from utils.replay_buffer import ReplayBuffer
from utils.rl_base_agent import RLBaseBidAgent
from utils.enums import ConfigParameters, ConfigSection, StateKeys


class TD3BidAgent(RLBaseBidAgent):
    """
    Agent that makes bids using Twin Delayed Deep Deterministic Policy Gradient (TD3) reinforcement learning algorithm
    (RL).

    This agent uses RL to determine optimal bid prices based on real-time auction data. It maintains internal state
    information such as remaining budget, time steps, and performance metrics like click-through rates (CTR), win
    rates (WR), and cost per mille (CPM). The agent updates its strategy based on the rewards received from successful
    bids and adjusts the bidding multiplier accordingly.

    Attributes:
        base_bid (int): base bid price from the contract
        budget (int): total budget for the AD campaign.
        seed (int): seed for random number generators to ensure reproducibility
        state_size (int): size of the state vector used by the TD3 agent, loaded from config.cfg
        batch_size (int): batch size for sampling experiences from the replay buffer, loaded from config.cfg
        replay_buffer_size (int): maximum size of the replay buffer, loaded from config.cfg
        action_size (int): size of the action space (continuous, i.e 1), representing a multiplier for the base bid, loaded from config.cfg
        td3_agent (TD3): TD3 agent himself
        memory (ReplayBuffer): Replay buffer to store experiences for training the TD3 agent
        ctl_lambda (float): control parameter that adjusts the bid multiplier
        day (StateKeys): current weekday, used to track episode boundaries
        hour (StateKeys): current hour of the day, used to track time step transitions
        pCTR (StateKeys): current pCTR of the day, used for bid scaling
        td3_state (np.ndarray): current state vector used for training the TD3 agent
        td3_action (float): last action taken by the TD3 agent
        td3_reward (float): reward received for the last action

    Methods:
        act: determines the bid price based on the current state and updates internal states
        _load_config: parses the config.cfg configuration file
        _reset_episode: reset the state when episode changes

    The agent loads configuration parameters from `config.cfg` file to initialize key attributes.
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
        Initialize TD3 agent with networks and exploration parameters.

        Args:
            base_bid (int): bid price from contract
            budget (int): AD campaing's budget
            seed (int): state of a random function
        """
        super().__init__(base_bid, budget, seed)

        self._reset_episode()

        self.td3_agent = TD3(
            state_size=self.state_size, 
            action_size=self.action_size, 
            seed=seed
        )

        self.memory = ReplayBuffer(
            buffer_size=self.replay_buffer_size, 
            batch_size=self.batch_size, 
            seed=seed
        )

        self.td3_state = None
        self.td3_action = 0.0
        self.td3_reward = 0

    def _reset_episode(self) -> None:
        """
        Reset episode with TD3-specific components.
        """
        super()._reset_episode()

        self.td3_state = None
        self.td3_action = 0.0
        self.td3_reward = 0

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

            td3_next_state = self._get_state()

            if self.td3_state is not None:
                self.memory.add(
                    self.td3_state,
                    [self.td3_action],
                    self.td3_reward,
                    td3_next_state,
                    episode_done,
                )

            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.td3_agent.learn(experiences)

            action_scale = self.td3_agent.act(td3_next_state.reshape(1, -1))
            self.td3_action = np.clip(action_scale[0], -1.0, 1.0)

            self.ctl_lambda = np.clip(self.ctl_lambda * (1 + self.td3_action * 0.1), 0.5, 2.0)
            self.hour = state.get(StateKeys.HOUR.value)

            self.td3_state = td3_next_state
            self.td3_reward = reward

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

