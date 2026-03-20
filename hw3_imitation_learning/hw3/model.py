"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int,
        depth: int,
        p: float,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=p)

        for i in range(depth):
            in_dim = state_dim if i == 0 else d_model
            out_dim = chunk_size * action_dim if i == depth - 1 else d_model
            self.layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                self.layers.append(nn.ReLU())
                self.layers.append(self.dropout)
        

    def forward(
        self, state: torch.Tensor
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""

        x = state
        for layer in self.layers:
            x = layer(x)
        return x.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_chunk = self.forward(state)
        loss = nn.functional.mse_loss(pred_chunk, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(state)
        


# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int, d_model: int, depth: int, p: float = 0.05):
        super().__init__(state_dim, action_dim, chunk_size)

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=p)

        for i in range(depth):
            in_dim = state_dim if i == 0 else d_model
            out_dim = chunk_size * action_dim if i == depth - 1 else d_model
            self.layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                self.layers.append(nn.ReLU())
                self.layers.append(self.dropout)
        

    def forward(
        self, state: torch.Tensor
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""

        x = state
        for layer in self.layers:
            x = layer(x)
        return x.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_chunk = self.forward(state)
        loss = nn.functional.mse_loss(pred_chunk, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(state)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    d_model: int,
    depth: int,
    p: float = 0.1,
    **kwargs,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            p=p
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            p=p
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
