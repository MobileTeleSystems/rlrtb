import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Critic's Q-value approximator network for DDPG algorithm.

    Parameters:
        state_size (int): dimension of input state space
        action_size (int): dimension of input action space
        seed (int): random seed for reproducibility
        fc1_layer (int, optional): number of neurons in first hidden layer. Default: 256
        fc2_layer (int, optional): number of neurons in second hidden layer. Default: 128

    Attributes:
        fc1 (nn.Linear): first fully connected layer for state processing
        bn1 (nn.BatchNorm1d): batch normalization layer after first fully connected
        fc2 (nn.Linear): second fully connected layer after action concatenation
        fc3 (nn.Linear): final output layer for Q-value estimation

    Methods:
        forward: computes Q-values for given state-action pairs
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 seed: int,
                 fc1_layer: int = 256,
                 fc2_layer: int = 128):
        """
        Initialization and model building.

        Args:
            state_size (int): each state dimension
            action_size (int): each action dimension
            seed (int): state of a random function
            fc1_layer (int): first hidden layer size
            fc2_layer (int): second hidden layer size
        """
        super(Critic, self).__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.fc1 = nn.Linear(state_size, fc1_layer)
        self.bn1 = nn.BatchNorm1d(fc1_layer)
        self.fc2 = nn.Linear(fc1_layer + action_size, fc2_layer)
        self.fc3 = nn.Linear(fc2_layer, 1)

    def forward(self, 
                state: torch.Tensor, 
                action: torch.Tensor) -> torch.Tensor:
        """
        Mapping the state-action pairs to Q-values.

        Args:
            state (torch.Tensor): current state
            action (torch.Tensor): current action
        Return:
            (torch.Tensor): Q-value
        """
        xs = self.bn1(F.relu(self.fc1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))

        return self.fc3(x)
