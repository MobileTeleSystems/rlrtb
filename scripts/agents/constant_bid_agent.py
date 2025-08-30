from utils.simple_base_agent import SimpleBaseAgent


class ConstantBidAgent(SimpleBaseAgent):    
    """
    Agent that makes a bid equal to the AD contract price.

    Implements basic bidding behavior for:
    - baseline performance benchmarking
    - campaign budget tracking
    - track campaign performance metrics
    - simple budget-constrained participation

    Parameters:
        base_bid (int): fixed bid value from AD contract
        budget (int): remaining campaign budget

    Methods:
        act: generate constant bid response
    """
    def act(self, 
            _state: list, 
            _reward: float, 
            _cost: float) -> int:
        """
        Generate constant bid response ignoring market conditions.

        Args:
            _state (list): current state (unused in this implementation)
            _reward (float): reward for completed action (unused in this implementation)
            _cost (float): bid price or market price according to auction type (unused in this implementation)
        Returns:
            (int): bid price
        """
        return self.base_bid