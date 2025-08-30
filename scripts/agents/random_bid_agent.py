import random

from utils.simple_base_agent import SimpleBaseAgent


class RandomBidAgent(SimpleBaseAgent):
    """
    Agent that makes a random bid in the [min_cpm; base_bid] interval.

    Implements basic bidding behavior for:
    - generating random bids within a specified range
    - campaign budget tracking
    - tracking campaign performance metrics
    - simple budget-constrained participation

    Parameters:
        base_bid (int): fixed bid value from AD contract
        budget (int): remaining campaign budget
        seed (int): seed for random number generators to ensure reproducibility
        min_cpm (int): minimum possible CPM

    Methods:
        act: generate a random bid response, but not less than min_cpm
    """
    def __init__(self, 
                 base_bid: int, 
                 budget: int, 
                 seed: int, 
                 min_cpm: int):
        """
        Initialize of an Agent object.

        Args:
            base_bid (int): bid price from contract
            budget (int): AD campaing's budget
            seed (int): state of a random function
            min_cpm (int): minimum possible CPM
        """
        super().__init__(base_bid, budget)
        
        random.seed(seed)
        
        self.min_cpm = min_cpm

    def act(self, 
            _state: list, 
            _reward: float, 
            _cost: float) -> int:
        """
        Taking action on every bid request.

        Args:
            state (list): current state (unused in this implementation)
            reward (float): reward for completed action (unused in this implementation)
            cost (float): bid price or market price according to auction type (unused in this implementation)
        Returns:
            (int): scaled bid price
        """
        return random.randint(self.min_cpm, self.base_bid)