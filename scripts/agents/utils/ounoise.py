import copy
import numpy as np


class OUNoise:
    """
    Ornstein-Uhlenbeck process for generating correlated noise.

    Used in DDPG for exploration in continuous action spaces. Generates noise with momentum where subsequent values are
    correlated with previous ones.

    Parameters:
        size (int): dimension of noise vector
        seed (int): random seed for reproducibility
        mu (float, optional): mean value of the process. Default: 0.0
        theta (float, optional): drift coefficient (speed of mean reversion). Default: 0.15
        sigma (float, optional): diffusion coefficient (noise magnitude). Default: 0.2

    Attributes:
        mu (np.ndarray): mean value vector
        theta (float): drift coefficient
        sigma (float): diffusion coefficient
        state (np.ndarray): current noise state

    Methods:
        reset: reset noise process to mean
        sample: generate next noise sample
    """

    def __init__(self,
                 size: int,
                 seed: int,
                 mu: float = 0.0,
                 theta: float = 0.15,
                 sigma: float = 0.2):
        """
        Initialize parameters and noise process.
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma

        np.random.seed(seed)

        self.reset()

    def reset(self) -> None:
        """
        Reset the internal state (noise) to mean (mu).
        """
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """
        Update internal state and return it as a noise sample.

        dx = θ * (μ - x) + σ * dW
        dW is white noise (normal distribution)

        Returns:
            (np.ndarray) - noised state
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx

        return self.state
