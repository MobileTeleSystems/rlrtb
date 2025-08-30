import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class DoubleQNetwork(nn.Module):
    """
    Critic model that estimates the Q-values using two separate networks Q1 and Q2 for bias reduction.

    Implements two separate Q-networks (Q1, Q2) that:
    - take state-action pairs as input
    - output Q-value estimates
    - share same architecture but independent parameters

    Parameters:
        state_size (int): dimension of input state vector
        action_size (int): dimension of action space
        seed (int): random seed for reproducibility
        hidden_size (int, optional): number of units in hidden layers. Default: 256

    Methods:
        forward: computes Q-values for given state-action pairs using both Q-networks
        q1: computes Q-values for given state-action pairs using single Q-network
    """

    def __init__(self, 
                 state_size: int, 
                 action_size: int, 
                 seed: int, 
                 hidden_size: int = 256):
        """
        Initialization and model building.

        Args:
            state_size (int): dimension of input state vector
            action_size (int): dimension of action space
            seed (int): random seed for reproducibility
            hidden_size (int, optional): number of units in hidden layers
        """
        super(DoubleQNetwork, self).__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.fc1_q1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2_q1 = nn.Linear(hidden_size, hidden_size)
        self.fc3_q1 = nn.Linear(hidden_size, 1)

        self.fc1_q2 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2_q2 = nn.Linear(hidden_size, hidden_size)
        self.fc3_q2 = nn.Linear(hidden_size, 1)

    def forward(self, 
                state: torch.Tensor, 
                action: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Computes the Q-values for a given state-action pair using both Q1 and Q2 networks.

        Args:
            state (torch.Tensor): current state tensor of shape (batch_size, state_size)
            action (torch.Tensor): action tensor of shape (batch_size, action_size)

        Returns:
            (Tuple[torch.Tensor]): tuple containing two tensors with Q-values from Q1 and Q2
        """
        x_q1 = torch.cat([state, action], 1)
        xq1 = F.relu(self.fc1_q1(x_q1))
        xq1 = F.relu(self.fc2_q1(xq1))
        q1 = self.fc3_q1(xq1)

        x_q2 = torch.cat([state, action], 1)
        xq2 = F.relu(self.fc1_q2(x_q2))
        xq2 = F.relu(self.fc2_q2(xq2))
        q2 = self.fc3_q2(xq2)

        return q1, q2

    def q1(self, 
           state: torch.Tensor, 
           action: torch.Tensor) -> torch.Tensor:
        """
        Computes the Q-value for a given state-action pair using only the Q1 network.
        Used for policy updates where minimum of both Q-values is not required.

        Args:
            state (torch.Tensor): current state tensor of shape (batch_size, state_size)
            action (torch.Tensor): action tensor of shape (batch_size, action_size)

        Returns:
            q1 (torch.Tensor): tensor with Q-values from Q1
        """
        x_q1 = torch.cat([state, action], 1)
        xq1 = F.relu(self.fc1_q1(x_q1))
        xq1 = F.relu(self.fc2_q1(xq1))
        q1 = self.fc3_q1(xq1)

        return q1
