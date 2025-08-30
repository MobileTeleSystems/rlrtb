import random

from utils.simple_base_agent import SimpleBaseAgent


class RandomParticipationBidAgent(SimpleBaseAgent):
    """
    Agent that makes a bid equal to the contract price, but sometimes exits the auction
    or enters randomly.

    Implements:
    - random participation in auctions based on a given probability
    - fixed bidding strategy where the bid is always equal to the base bid when participating
    - campaign budget tracking
    - tracking campaign performance metrics

    Parameters:
        base_bid (int): fixed bid value from AD contract
        budget (int): remaining campaign budget
        seed (int): state of a random function
        participation_probability (float, optional): probability of participating in an auction (0 to 1). Default: 0.4

    Methods:
        act: sometimes generate constant bid response
    """
    def __init__(self, 
                 base_bid: int, 
                 budget: int, 
                 seed: int, 
                 participation_probability: float = 0.4):
        """
        Initialize of an Agent object.

        Args:
            base_bid (int): bid price from contract
            budget (int): AD campaign's budget
            seed (int): state of a random function
            participation_probability (float): Probability of participating in an auction (0 to 1).
        """
        super().__init__(base_bid, budget)
        
        random.seed(seed)
        
        self.participation_probability = participation_probability

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
        return self.base_bid if random.random() < self.participation_probability else 0