import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

LOG_STD_MAX = 2  # upper bound for log standard deviation
LOG_STD_MIN = -20  # lower bound for log standard deviation


class GaussianPolicy(nn.Module):
    """
    Stochastic policy network for SAC agent using Gaussian action distribution with tanh squashing.

    This policy network takes in the current state and produces the mean and log standard deviation of a Gaussian
    distribution from which actions are sampled. The reparameterization trick is used to ensure that the gradient can
    flow through the sampling process.

    Implements:
    - two hidden layers with ReLU activation
    - state-action mapping with mean and log std outputs
    - reparameterization trick for differentiable sampling
    - action squashing to [-1, 1] range via tanh
    - log probability calculation with tanh correction

    Parameters:
        state_size (int): dimension of input state vector
        action_size (int): dimension of action space
        seed (int): random seed for reproducibility
        hidden_size (int, optional): number of units in hidden layers. Default: 256

    Methods:
        forward: computes mean and log std of the Gaussian distribution over actions
        sample: sample an action from the Gaussian distribution and computes its log probability
    """

    def __init__(self, 
                 state_size: int, 
                 action_size: int, 
                 seed: int, 
                 hidden_size: int = 256):
        super(GaussianPolicy, self).__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, 
                state: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Computes the mean and log standard deviation of the Gaussian distribution over actions.

        Implements:
        1. Gaussian sampling with reparameterization trick
        2. tanh squashing for bounded actions
        3. log probability correction for tanh transformation

        Args:
            state (torch.Tensor): current state

        Returns:
            (Tuple): a tuple containing two tensors:
                mean (torch.Tensor): mean of the Gaussian distribution
                log_std (torch.Tensor): clamped log standard deviation of the Gaussian distribution
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, 
               state: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Samples an action from the Gaussian distribution and computes its log probability.

        Args:
            state (torch.Tensor): current state

        Returns:
            (Tuple): a tuple containing two tensors:
                action (torch.Tensor): sampled action tensor after applying the tanh function
                log_prob (torch.Tensor): log probability of the sampled action, adjusted for the tanh transformation
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()  # reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)  # squashing action
        action = y_t

        log_prob = normal.log_prob(x_t)

        # enforcing action bound
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)  # Jacobian correction
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob
