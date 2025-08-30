from collections import deque

from utils.enums import StateKeys
from utils.simple_base_agent import SimpleBaseAgent
    

class LinearBidAgent(SimpleBaseAgent):
    """
    An agent designed to make bids in an auction environment based on the contract price, scaled by the ratio of the
    probability of a click for the current bid request to the average click probability.

    Implements advanced bidding behavior for:
    - dynamic bid adjustment based on recent click probabilities
    - campaign budget tracking
    - tracking campaign performance metrics
    - budget-constrained participation with adaptive scaling

    Parameters:
        base_bid (int): fixed bid value from AD contract
        budget (int): remaining campaign budget
        win_size (int, optional): the size of the sliding window used to calculate the average click probability. Default: 10000

    Attributes:
        click_probs (deque): a deque that stores the recent click probabilities observed

    Methods:
        act: generate a scaled bid response
        update: record auction outcomes and update budget
    """
    def __init__(self, 
                 base_bid: int, 
                 budget: int, 
                 win_size: int = 10000):
        """
        Initialize of an Agent object.

        Args:
            base_bid (int): bid price from contract
            budget (int): AD campaing's budget
            win_size (int): window width for calculating the bid coefficient
        """
        super().__init__(base_bid, budget)
        
        self.click_probs = deque(maxlen=win_size)

    def act(self, 
            state: list, 
            _reward: float, 
            _cost: float) -> float:
        """
        Taking action on every bid request.

        Args:
            state (dict): current state, should contain 'pCTR' and 'click'
            _reward (float): reward for completed action (unused in this implementation)
            _cost (float): bid price or market price according to auction type (unused in this implementation)
        Returns:
            action (float): scaled bid price
        """
        bid_coef = sum(self.click_probs) / len(self.click_probs) if self.click_probs else 1.0

        return self.base_bid * min(state.get(StateKeys.PCTR.value, 0.0) / bid_coef, 1)

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
        super().update(observation, reward, cost)
        
        self.click_probs.append(observation.get(StateKeys.PCTR.value, 0.0))