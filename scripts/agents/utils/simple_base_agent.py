from abc import ABC, abstractmethod

from utils.enums import StateKeys


class SimpleBaseAgent(ABC):
    """
    Base class for bid optimization agents in real-time advertising auctions. Provides common functionality and 
    interface for all agent implementations.

    Defines:
    - core attributes for budget tracking and performance metrics
    - standardized methods for action selection and state updates
    - common budget exhaustion check
    - basic infrastructure for reinforcement learning agents
    
    Parameters:
        base_bid (int): fixed bid value from AD contract
        budget (int): remaining campaign budget

    Attributes:
        total_bids (int): lifetime bid participation count
        total_wins (int): lifetime auction wins count
        total_rewards (float): cumulative rewards from clicks
        total_clicks (int): total clicks obtained
        total_budget_spend (float): total expenditure across auctions

    Methods:
        act: abstract method to select an action (implementation dependent)
        update: record auction outcomes and update budget
        done: check if the budget is exhausted
    """
    def __init__(self, 
                 base_bid: int, 
                 budget: int):
        """
        Initialize of an Agent object.

        Args:
            base_bid (int): bid price from contract
            budget (int): AD campaing's budget
        """
        self.base_bid = base_bid
        self.budget = budget
        
        self.total_bids = 0
        self.total_wins = 0
        self.total_rewards = 0.0
        self.total_clicks = 0
        self.total_budget_spend = 0

    @abstractmethod
    def act(self, 
            state: dict, 
            reward: float, 
            cost: float) -> float:
        """
        Generate bid response for auction participation.
        Must be implemented by all subclasses.
        
        Args:
            state (list): current state (implementation-dependent usage)
            reward (float): reward for completed action (implementation-dependent usage)
            cost (float): bid price or market price according to auction type (implementation-dependent usage)
        Returns:
            action (int): scaled bid price for next auction
        """
        pass

    def update(self, 
               observation: dict, 
               reward: float, 
               cost: float) -> None:
        """
        Updates the agent's internal state based on the outcome of the auction.

        Args:
            observation (dict): the observation for the current request.
            reward (float): the reward received for the bid.
            cost (float): the cost incurred for the bid.
        """
        self.total_bids += 1

        if cost > 0:
            self.total_budget_spend += cost
            self.total_wins += 1
            self.total_rewards += reward
            self.budget -= cost

        self.total_clicks += observation.get(StateKeys.CLICK.value, 0) if reward else 0

    def done(self) -> bool:
        """
        Returns bool flag if there is no action left.

        Returns:
            (bool): there is no actions left
        """
        return self.budget <= self.base_bid