from utils.simple_base_agent import SimpleBaseAgent


class BudgetPacingAgent(SimpleBaseAgent):
    """
    Adaptive bidding agent with budget pacing strategy for optimal campaign spending.

    Implements time-dependent bid adjustment to:
    - distribute budget evenly across total_steps
    - dynamically scale bids based on spending progress
    - maintain participation rate through bid multipliers
    - track campaign performance metrics

    Parameters:
        base_bid (int): fixed bid value from AD contract
        budget (int): remaining campaign budget
        total_steps (int): total planned auction participation steps
        bid_multiplier (float, optional): bid scaling factor for aggressive pacing. Default: 1.1

    Attributes:
        current_step (int): current progress through auction timeline

    Methods:
        act: generate paced bid based on spending trajectory
        update: record auction outcomes and update pacing state
    """
    def __init__(self, 
                 base_bid: int, 
                 budget: int, 
                 total_steps: int, 
                 bid_multiplier: float = 1.1):
        """
        Initialize of an Agent object.

        Args:
            base_bid (int): bid price from contract
            budget (int): AD campaing's budget
            total_steps (int): total amount of bids
            bid_multiplier (float): scaling factor
        """
        super().__init__(base_bid, budget)
        
        self.total_steps = total_steps
        self.bid_multiplier = bid_multiplier

        self.current_step = 0

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
            self.current_step += 1

    def act(self, 
            _state: list, 
            _reward: float, 
            _cost: float) -> int:
        """
        Taking action on every bid request.

        Args:
            _state (list): current state (unused in this implementation)
            _reward (float): reward for completed action (unused in this implementation)
            _cost (float): bid price or market price according to auction type (unused in this implementation)
        Returns:
            action (int): scaled bid price or 0 if the agent doesn't participate.
        """
        target_spend = (self.current_step / self.total_steps) * self.budget

        return int(self.base_bid * self.bid_multiplier) if self.total_budget_spend < target_spend else self.base_bid