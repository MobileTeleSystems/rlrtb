import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.noisy_linear import NoisyLinear
from utils.enums import LinearLayerType


class DuelingQNetwork(nn.Module):
    """
    DuelingDQN uses Dueling Q-head in order to separate Q to an advantage (A) and value (V) stream. Adding this type of
    structure to the network head allows the network to better diffentiate actions from one another, and improves the
    learning as we start learning her (s, a) even if only a single action has been taken in this state.

    Implements:
    - two parallel hidden layers for value (V) and advantage (A) estimation
    - final Q-values calculated as: Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
    - supports both regular and noisy linear layers

    Based on: "Dueling Network Architectures for Deep Reinforcement Learning"
    https://arxiv.org/abs/1511.06581

    Parameters:
        state_size (int): dimension of input state space
        action_size (int): dimension of output action space
        seed (int): random seed for reproducibility
        layer_type (LinearLayerType, optional): layer variant ('regular' or 'noisy'). Default: 'regular'
        fc_layer (int, optional): neuron count in the hidden layer. Default: 64

    Attributes:
        fc1_adv (nn.Linear/NoisyLinear): Advantage stream first layer
        fc1_val (nn.Linear/NoisyLinear): Value stream first layer
        fc2_adv (nn.Linear/NoisyLinear): Advantage stream output layer
        fc2_val (nn.Linear/NoisyLinear): Value stream output layer

    Methods:
        forward: computes Q-values for given states
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 seed: int,
                 layer_type: LinearLayerType = LinearLayerType.REGULAR.value,
                 fc_layer: int = 64):
        """
        Initialization and model building.

        Args:
            state_size (int): dimension of input state space
            action_size (int): dimension of output action space
            seed (int): state of a random function
            layer_type (LinearLayerType): linear layer type
            fc1_layer (int): first hidden layer size
        """
        super(DuelingQNetwork, self).__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.action_size = action_size
        self.layer_type = nn.Linear if layer_type == LinearLayerType.REGULAR.value else NoisyLinear

        self.fc1_adv = self.layer_type(state_size, fc_layer)
        self.fc1_val = self.layer_type(state_size, fc_layer)

        self.fc2_adv = self.layer_type(fc_layer, action_size)
        self.fc2_val = self.layer_type(fc_layer, 1)

    def forward(self, 
                state: list) -> float:
        """
        Forward pass through dueling networkю

        Args:
            state (list): current state
        Return:
            q (float): Q-value
        """
        adv = F.relu(self.fc1_adv(state))
        val = F.relu(self.fc1_val(state))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(state.size(0), self.action_size)

        q = val + (adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.action_size))

        return q
