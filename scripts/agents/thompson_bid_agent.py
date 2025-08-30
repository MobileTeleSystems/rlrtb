from scipy.stats import beta

from utils.enums import StateKeys
from utils.simple_base_agent import SimpleBaseAgent


class ThompsonSamplingBidAgent(SimpleBaseAgent):
    """
    Agent that makes a bid equal to the contract price, scaled by expected click-through rate (pCTR) using Thompson
    Sampling.

    Implements:
    - bidding strategy based on the Thompson Sampling algorithm
    - campaign budget tracking
    - tracking campaign performance metrics
    - dynamic bid scaling based on sampled and observed click-through rates

    Parameters:
        base_bid (int): fixed bid value from AD contract
        budget (int): remaining campaign budget

    Attributes:
        model_weight (float): weight for combining sampled pCTR with observed pCTR

    Methods:
        act: generate a scaled bid response
    """
    def __init__(self, 
                 base_bid: int, 
                 budget: int):
        """
        Initialize of an Agent object.

        Args:
            base_bid (int): bid price from contract
            budget (int): AD campaign's budget
        """
        super().__init__(base_bid, budget)
        
        self.model_weight = 0.5

    def act(self, 
            state: dict, 
            _reward: float, 
            _cost: float) -> float:
        """
        Make a bid decision based on Thompson Sampling.

        Args:
            state (dict): current state, should contain pCTR and click
            _reward (float): reward for completed action (unused in this implementation)
            _cost (float): bid price or market price according to auction type (unused in this implementation)
        Returns:
            action (float): scaled bid price
        """
        alpha = self.total_clicks + 1
        beta_ = self.total_bids - self.total_clicks + 1
        sampled_pctr = beta.rvs(alpha, beta_)

        combined_pctr = (1 - self.model_weight) * sampled_pctr + self.model_weight * state.get(StateKeys.PCTR.value, 0.0)

        return self.base_bid * min(combined_pctr, 1)