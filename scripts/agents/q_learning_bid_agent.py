import math
import random
import numpy as np

from utils.simple_base_agent import SimpleBaseAgent
from utils.enums import StateKeys, ExplorationStrategy


class QLearningBidAgent(SimpleBaseAgent):
    """
    Q-learning agent for optimizing bids in real-time advertising auctions.

    Implements a Q-learning algorithm with multiple exploration strategies to adaptively choose bid prices based on
    predicted CTR (pCTR), campaign budget, and market conditions. Supports various exploration strategies including
    ε-greedy, UCB, Boltzmann, and pursuit methods.

    Implements:
    - adaptive bid optimization using Q-learning algorithm
    - multiple exploration-exploitation strategies (ε-greedy, UCB, Boltzmann, pursuit)
    - state discretization for pCTR values
    - budget-constrained action selection
    - real-time Q-value updates with experience replay
    - exploration rate annealing for ε-greedy strategy
    - non-uniform action selection probabilities
    - campaign budget tracking
    - tracking campaign performance metrics

    Parameters:
        base_bid (int): fixed bid value from AD contract
        budget (int): remaining campaign budget
        seed (int): state of a random function
        learning_rate (float, optional): learning rate for Q-value updates. Default: 0.1
        discount_factor (float, optional): Discount factor for future rewards. Default: 0.5
        exploration_rate (float, optional): Initial exploration probability (ε). Default: 1.0
        exploration_decay_rate (float, optional): Decay rate for exploration probability. Default: 5e-4
        min_exploration_rate (float, optional): Minimum exploration probability. Default: 0.01
        num_pctr_bins (int, optional): Number of bins for discretizing pCTR values. Default: 25
        num_possible_bids (int, optional): Number of discrete bid options. Default: 10
        initial_bid_multiplier_range (tuple, optional): Multiplier range for generating bid options. Default: (0.1, 1.0)
        exploration_strategy (ExplorationStrategy, optional): Exploration strategy ('epsilon_greedy', 'ucb', 'boltzmann', 'pursuit'). 
            Default: ExplorationStrategy.EPSILON_GREEDY.value
        ucb_constant (float, optional): Exploration constant for UCB. Default: 1.0
        boltzmann_temperature (float, optional): Temperature parameter for Boltzmann exploration. Default: 1.0
        pursuit_learning_rate (float, optional): Learning rate for pursuit strategy. Default: 0.1

    Attributes:
        q_table (numpy.ndarray): Q-value table of shape [num_pctr_bins x num_possible_bids]
        possible_bids (numpy.ndarray): array of possible bid values

    Methods:
        act: generate a scaled bid response
        update: record auction outcomes and update budget
        _generate_possible_bids: generate a list of possible bid amounts 
        _discretize_state: discretize the pCTR value to a bin index
        _get_action_index: selects an action index (bid index) based on the exploration strategy
    """
    def __init__(self,
                 base_bid: int,
                 budget: int,
                 seed: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.5,
                 exploration_rate: float = 1.0,
                 exploration_decay_rate: float = 5e-4,
                 min_exploration_rate: float = 0.01,
                 num_pctr_bins: int = 25,
                 num_possible_bids: int = 10,
                 initial_bid_multiplier_range: tuple = (0.1, 1.0),
                 exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY.value,
                 ucb_constant: float = 1.0,
                 boltzmann_temperature: float = 1.0,
                 pursuit_learning_rate: float = 0.1):
        """
        Initialize a Q-learning Agent object.

        Args:
            base_bid (int): bid price from contract
            budget (int): AD campaign's budget
            seed (int): state of a random function
            learning_rate (float): learning rate for Q-learning update
            discount_factor (float): discount factor for future rewards
            exploration_rate (float): initial exploration rate (for epsilon-greedy)
            exploration_decay_rate (float): rate at which exploration rate decays (for epsilon-greedy)
            min_exploration_rate (float): minimum exploration rate (for epsilon-greedy)
            num_pctr_bins (int): number of bins to discretize pCTR values
            num_possible_bids (int): number of discrete bid values the agent can choose from
            initial_bid_multiplier_range (tuple): range of multipliers for generating possible bids.
            exploration_strategy (ExplorationStrategy): exploration strategy 'epsilon_greedy', 'ucb', 'boltzmann', or 'pursuit'
            ucb_constant (float): constant for UCB calculation
            boltzmann_temperature (float): temperature parameter for Boltzmann exploration
            pursuit_learning_rate (float): learning rate for Pursuit exploration
        """
        super().__init__(base_bid, budget)
        
        random.seed(seed)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.min_exploration_rate = min_exploration_rate
        self.num_pctr_bins = num_pctr_bins
        self.num_possible_bids = num_possible_bids
        self.initial_bid_multiplier_range = initial_bid_multiplier_range
        self.exploration_strategy = exploration_strategy

        self.ucb_constant = ucb_constant
        self.boltzmann_temperature = boltzmann_temperature
        self.pursuit_learning_rate = pursuit_learning_rate

        self.pctr_bins = np.linspace(0, 1, self.num_pctr_bins + 1)[1:-1]
        self.possible_bids = self._generate_possible_bids()
        self.q_table = np.zeros((self.num_pctr_bins, len(self.possible_bids)))

        self.last_state = None
        self.last_action_index = None

        # UCB
        self.state_action_visits = np.zeros((self.num_pctr_bins, len(self.possible_bids)))
        self.state_visits = np.zeros(self.num_pctr_bins)

        # Pursuit exploration
        self.target_policy = np.ones((self.num_pctr_bins, len(self.possible_bids))) / len(self.possible_bids)

    def _generate_possible_bids(self) -> list:
        """
        Generate a list of possible bid amounts based on the initial bid multiplier range.

        Returns:
            (list): A list of possible bid amounts.
        """
        low = self.initial_bid_multiplier_range[0] * self.base_bid
        high = self.initial_bid_multiplier_range[1] * self.base_bid

        return np.linspace(low, high, self.num_possible_bids)
    
    def _discretize_state(self, 
                          state: dict) -> int:
        """
        Discretize the PCTR value to a bin index.

        Args:
            state (dict): dictionary containing the current state, e.g., {'pctr': 0.15}.
        Returns:
            (int): the index of the bin corresponding to the pCTR value.
        """
        pctr = state.get(StateKeys.PCTR.value, 0.0)

        return np.digitize(pctr, self.pctr_bins)
    
    def _get_action_index(self, 
                          state_index: int) -> int:
        """
        Selects an action index (bid index) based on the exploration strategy, considering the available budget.

        Args:
            state_index (int): discretized state index.
        Returns:
            (int): index of the chosen bid in the possible_bids list, or None if no affordable bids are available.
        """
        possible_action_indices = [
            i for i, bid in enumerate(self.possible_bids) if bid <= self.budget
        ]

        if not possible_action_indices:
            return None

        if self.exploration_strategy == ExplorationStrategy.EPSILON_GREEDY.value:
            if random.random() < self.exploration_rate:
                return random.choice(possible_action_indices)
            else:
                q_values = self.q_table[state_index, possible_action_indices]

                return possible_action_indices[np.argmax(q_values)]
        elif self.exploration_strategy == ExplorationStrategy.UCB.value:
            q_values = self.q_table[state_index, possible_action_indices]
            ucb_values = np.zeros_like(q_values, dtype=float)

            for idx, action_index in enumerate(possible_action_indices):
                visits = self.state_action_visits[state_index, action_index]
                total_state_visits = self.state_visits[state_index]

                if total_state_visits > 0:
                    ucb_values[idx] = q_values[idx] + self.ucb_constant * math.sqrt(math.log(total_state_visits) / (visits + 1e-6))
                else:
                    ucb_values[idx] = float("inf")

            return possible_action_indices[np.argmax(ucb_values)]

        elif self.exploration_strategy == ExplorationStrategy.BOLTZMANN.value:
            q_values = self.q_table[state_index, possible_action_indices]

            exp_q_values = np.exp(q_values / self.boltzmann_temperature)
            probabilities = exp_q_values / np.sum(exp_q_values)

            return random.choices(
                possible_action_indices, 
                weights=probabilities, 
                k=1
            )[0]
        elif self.exploration_strategy == ExplorationStrategy.PURSUIT.value:
            probabilities = self.target_policy[state_index, possible_action_indices]

            return random.choices(
                possible_action_indices, 
                weights=probabilities, 
                k=1
            )[0]
        else:
            raise ValueError(
                f"Unknown exploration strategy: {self.exploration_strategy}"
            )

    def act(self, 
            state: dict, 
            _reward: float, 
            _cost: float) -> float:
        """
        Taking action on every bid request using Q-learning.

        Args:
            state (dict): current state
            _reward (float): reward for completed action (unused in this implementation)
            _cost (float): bid price or market price according to auction type (unused in this implementation)
        Returns:
            action (float): the bid price
        """
        state_index = self._discretize_state(state)
        action_index = self._get_action_index(state_index)

        if action_index is None:
            return 0.0

        self.last_state = state
        self.last_action_index = action_index
        action = self.possible_bids[action_index]

        return action

    def update(self, 
               observation: dict, 
               reward: float, 
               cost: float) -> None
        super().update(observation, reward, cost)

        if self.last_state is not None and self.last_action_index is not None:
            last_state_index = self._discretize_state(self.last_state)
            current_state_index = self._discretize_state(observation)

            best_next_action_value = np.max(self.q_table[current_state_index, :])
            old_value = self.q_table[last_state_index, self.last_action_index]
            td_error = reward + self.discount_factor * best_next_action_value - old_value
            self.q_table[last_state_index, self.last_action_index] += (self.learning_rate * td_error)

            if self.exploration_strategy == ExplorationStrategy.UCB.value:
                self.state_visits[last_state_index] += 1
                self.state_action_visits[last_state_index, self.last_action_index] += 1

            if self.exploration_strategy == ExplorationStrategy.PURSUIT.value:
                best_action_index = np.argmax(self.q_table[last_state_index])
                for i in range(len(self.possible_bids)):
                    if i == best_action_index:
                        self.target_policy[last_state_index, i] += self.pursuit_learning_rate * (1 - self.target_policy[last_state_index, i])
                    else:
                        self.target_policy[last_state_index, i] += self.pursuit_learning_rate * (0 - self.target_policy[last_state_index, i])

        if self.exploration_strategy == ExplorationStrategy.EPSILON_GREEDY.value:
            self.exploration_rate = max(self.min_exploration_rate, np.exp(-self.exploration_decay_rate * self.total_bids))