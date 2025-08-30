import numpy as np

from collections import deque

from utils.enums import StateKeys
from utils.simple_base_agent import SimpleBaseAgent


class UCBBidAgent(SimpleBaseAgent):
    """
    Agent that makes a bid equal to the contract price, scaled by expected click-through rate (pCTR) using Upper
    Confidence Bound (UCB).

    Implements:
    - bidding strategy based on the UCB algorithm
    - tracking of pCTR values over a sliding window
    - campaign budget tracking
    - tracking campaign performance metrics
    - dynamic bid scaling based on observed click-through rates

    Parameters:
        base_bid (int): fixed bid value from AD contract
        budget (int): remaining campaign budget
        win_size (int, optional): the size of the sliding window used to calculate the average click probability. Default: 10000

    Attributes:
        click_probs (deque): a deque that stores the recent click probabilities observed
        sum_pctr (float): sum of pCTR values for calculating mean
        sum_pctr_sq (float): sum of squared pCTR values for variance calculation

    Methods:
        act: generate a sclaed bid response, based on UCB
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
            budget (int): AD campaign's budget
            win_size (int): window width for click probs
        """
        super().__init__(base_bid, budget)
        
        self.win_size = win_size
        self.click_probs = deque(maxlen=self.win_size)

        self.sum_pctr = 0.0
        self.sum_pctr_sq = 0.0

    def act(self, 
            state: dict, 
            _reward: float, 
            _cost: float) -> float:
        """
        Make a bid decision based on UCB.

        Args:
            state (dict): current state, should contain pCTR and click
            _reward (float): reward for completed action (unused in this implementation)
            _cost (float): bid price or market price according to auction type (unused in this implementation)
        Returns:
            action (float): scaled bid price
        """
        n = len(self.click_probs)

        if n == 0:
            bid_coef = 1.0
        else:
            mean_pctr = self.sum_pctr / n
            confidence = np.sqrt((2 * np.log(self.total_bids + 1)) / n)
            bid_coef = max(mean_pctr - confidence, 1e-5)

        return self.base_bid * min(state.get(StateKeys.PCTR.value, 0.0) / bid_coef, 1.0)

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

        pctr = observation.get(StateKeys.PCTR.value, 0.0)

        if len(self.click_probs) == self.win_size:
            old_pctr = self.click_probs.popleft()
            self.sum_pctr -= old_pctr
            self.sum_pctr_sq -= old_pctr ** 2

        self.click_probs.append(pctr)
        self.sum_pctr += pctr
        self.sum_pctr_sq += pctr ** 2