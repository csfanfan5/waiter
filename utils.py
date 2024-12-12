import argparse
import os

# from jaxtyping import Float
from torch import Tensor, nn

class DiscretePolicy(nn.Module):
    """A feedforward neural network for discrete action spaces."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )

    def forward(
        self, states: Float[Tensor, "B state_dim"]
    ) -> Float[Tensor, "B action_dim"]:
        """Returns the action distribution for each state in the batch."""
        logits = self.net(states)
        return logits.float()
