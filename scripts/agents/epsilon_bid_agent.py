import random

from utils.enums import StateKeys
from utils.simple_base_agent import SimpleBaseAgent
    

class EpsilonGreedyBidAgent(SimpleBaseAgent):
    """
    Adaptive bidding agent combining ε-greedy exploration with pCTR-based bid scaling.

    Implements hybrid bidding strategy:
    - baseline bid scaled by predicted click-through rate
    - ε probability of random bid exploration
    - adaptive bid coefficient based on historical pCTR
    - budget-aware participation tracking

    Parameters:
        base_bid (int): fixed bid value from AD contract
        budget (int): remaining campaign budget
        seed (int): state of a random function
        epsilon (float, optional): exploration probability [0, 1]. Default: 0.1

    Attributes:
        click_probs (list): historical pCTR values from won auctions

    Methods:
        act: generate ε-greedy bid with pCTR scaling
        update: record auction outcomes and update pCTR history
    """
    def __init__(self, 
                 base_bid: int, 
                 budget: int, 
                 seed: int, 
                 epsilon: float = 0.1):
        """
        Initialize of an Agent object.

        Args:
            base_bid (int): bid price from contract
            budget (int): AD campaign's budget
            seed (int): state of a random function
            epsilon (float): probability of taking a random action
        """
        super().__init__(base_bid, budget)
        random.seed(seed)

        self.epsilon = epsilon
        self.click_probs = []

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

        if cost > 0:
            self.click_probs.append(observation.get(StateKeys.PCTR.value, 0.0))

    def act(self, 
            state: dict, 
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

        if random.random() < self.epsilon:
            return self.base_bid * random.uniform(0, min(state.get(StateKeys.PCTR.value, 0.0) / bid_coef, 1))
        
        return self.base_bid * min(state.get(StateKeys.PCTR.value, 0.0) / bid_coef, 1)