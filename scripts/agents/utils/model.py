import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.enums import LinearLayerType
from utils.noisy_linear import NoisyLinear


class QNetwork(nn.Module):
    """
    Actor (policy) model.

    Parameters:
        state_size (int): dimension of input state space
        action_size (int): dimension of output action space
        seed (int): random seed for reproducibility
        layer_type (LinearLayerType, optional): layer variant ('regular' or 'noisy'). Default: 'regular'
        fc1_layer (int, optional): number of units in first hidden layer. Default: 64
        fc2_layer (int, optional): number of units in second hidden layer. Default: 32

    Attributes:
        fc1 (nn.Linear/NoisyLinear): first hidden layer
        fc2 (nn.Linear/NoisyLinear): secind hidden layer
        fc3 (nn.Linear/NoisyLinear): output hidden layer

    Methods:
        forward: computes Q-values for given states
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 seed: int,
                 layer_type: LinearLayerType = LinearLayerType.REGULAR.value,
                 fc1_layer: int = 256,
                 fc2_layer: int = 128):
        """
        Initialization and model building.

        Args:
            state_size (int): dimension of input state vector
            action_size (int): dimension of action space
            seed (int): state of a random function
            layer_type (LinearLayerType): layer type
            fc1_layer (int): first hidden layer size
            fc2_layer (int): second hidden layer size
        """
        super(QNetwork, self).__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.layer_type = nn.Linear if layer_type == LinearLayerType.REGULAR.value else NoisyLinear
        self.fc1 = self.layer_type(state_size, fc1_layer)
        self.fc2 = self.layer_type(fc1_layer, fc2_layer)
        self.fc3 = self.layer_type(fc2_layer, action_size)

    def forward(self, 
                state: torch.Tensor) -> torch.Tensor:
        """
        Computes the Q-values for a given state.

        Args:
            state (torch.Tensor): current state
        Return:
            (torch.Tensor): Q-value estimates for all actions
        """
        s = F.relu(self.fc1(state))
        s = F.relu(self.fc2(s))

        return self.fc3(s)
