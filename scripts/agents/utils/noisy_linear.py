import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for independent Gaussian noise.

    Implements noise injection capabilities for:
    - factorized Gaussian noise generation (parameter-space noise)
    - trainable mu and sigma parameters for weights/biases
    - independent noise samples for each layer instance
    - noise reset and scaling mechanisms
    - compatibility with standard linear layer operations

    Parameters:
        in_features (int): number of input features
        out_features (int): number of output features
        std_init (float, optional): initial standard deviation for noise parameters. Default: 0.5

    Attributes:
        weight_mu (nn.Parameter): mean values for weight matrix
        weight_sigma (nn.Parameter): learnable scale parameters for weight noise
        weight_epsilon (Buffer): noise buffer for weight matrix
        bias_mu (nn.Parameter): mean values for bias vector
        bias_sigma (nn.Parameter): learnable scale parameters for bias noise
        bias_epsilon (Buffer): noise buffer for bias vector

    Methods:
        reset_parameters: initializes mu and sigma parameters
        reset_noise: generates new noise values
        forward: performs noisy linear transformation
        _scale_noise: set scale to make noise

    https://arxiv.org/pdf/1706.10295.pdf
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """

    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 std_init: float = 0.5):
        """
        Initialization.

        Args:
            in_features (int): input size of linear module
            out_features (int): output size of linear module
            std_init (float): initial std value
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """
        Reset trainable network parameters (factorized gaussian noise) and initialize mu and sigma.
        """
        mu = 1 / math.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu, mu)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu, mu)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, 
                     size: int) -> torch.Tensor:
        """
        Set scale to make noise (factorized gaussian noise).

        Args:
            size (int): the size of the noise tensor to generate.

        Returns:
            (torch.Tensor): a tensor with scaled noise.
        """
        x = torch.randn(
            size=(size,), 
            device=self.weight_mu.device
        )

        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None
        """
        Generate new noise.
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass with noisy weights and biases.

        Args:
            x (torch.Tensor): the input tensor.

        Returns:
            (torch.Tensor): the output tensor after applying the linear transformation.
        """
        return F.linear(
            input=x,
            weight=self.weight_mu + self.weight_sigma * self.weight_epsilon,
            bias=self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
