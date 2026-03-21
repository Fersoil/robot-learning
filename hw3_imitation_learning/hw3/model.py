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

class ResidualBlock(nn.Module):
    """Single residual block: Linear -> LayerNorm -> ReLU -> Dropout, with skip."""

    def __init__(self, d_model: int, p: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(p=p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualMLP(nn.Module):
    """Input projection -> N residual blocks -> output head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int,
        depth: int,
        p: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([ResidualBlock(d_model, p) for _ in range(depth)])
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.relu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class ObstaclePolicy(BasePolicy):
    """MLP with obstacle-aware feature injection, residual blocks, and LayerNorm.

    Architecture:
      1. input_proj   : full state  -> d_model
      2. obstacle_proj: last 3 dims -> d_model, added to input_proj output
         This gives the obstacle signal a direct gradient path.
      3. N residual blocks (Linear -> LayerNorm -> ReLU -> Dropout + skip)
      4. Linear head  -> (B, chunk_size, action_dim)

    state_ee_xyz state_gripper "state_cube[:3]" state_obstacle (state_obstacle must be the LAST 3 dims)
    """

    OBSTACLE_DIM = 3

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

        self.input_proj    = nn.Linear(state_dim, d_model)
        self.obstacle_proj = nn.Linear(self.OBSTACLE_DIM, d_model)
        self.blocks        = nn.ModuleList([ResidualBlock(d_model, p) for _ in range(depth - 1)])
        self.head          = nn.Linear(d_model, chunk_size * action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        obstacle = state[..., -self.OBSTACLE_DIM:]       # (B, 3)
        x = self.input_proj(state) + self.obstacle_proj(obstacle)
        x = nn.functional.relu(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(self.forward(state), action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(state)


class MultiTaskPolicy(BasePolicy):
    """
    MLP with one-hot attention to target cube, relative position features, residual blocks, and LayerNorm.
    """

    MLP_INPUT_DIM = 7  # gripper(1) + ee_to_cube(3) + ee_to_goal(3)

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int,
        depth: int,
        p: float = 0.05,
        state_mean: torch.Tensor | None = None,
        state_std:  torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        # Stored as buffers: saved in checkpoint, moved to device automatically
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std",  state_std)

        self.mlp = ResidualMLP(
            input_dim=self.MLP_INPUT_DIM,
            output_dim=chunk_size * action_dim,
            d_model=d_model,
            depth=depth,
            p=p,
        )

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Dynamically register buffers if they are missing (e.g. older checkpoints)."""
        for key in ("state_mean", "state_std"):
            if key in state_dict and getattr(self, key, None) is None:
                self.register_buffer(key, state_dict[key])
        return super().load_state_dict(state_dict, *args, **kwargs)

    def _build_input(self, state: torch.Tensor) -> torch.Tensor:
        """Undo normalisation, apply one-hot attention, compute relative vectors."""
        B = state.shape[0]

        # Undo z-score so that one-hot values are crisp 0/1 and xyz in metres
        s = state * self.state_std + self.state_mean

        robot_xyz = s[:, :3]                                    # (B, 3)
        gripper   = s[:, 3:4]                                   # (B, 1)
        cubes     = s[:, 4:13].view(B, 3, 3)                   # (B, 3, 3)
        goal_one_hot = nn.functional.one_hot(
            s[:, 13:16].argmax(dim=1), num_classes=3
        ).float()                                               # (B, 3)
        goal_pos  = s[:, 16:19]                                 # (B, 3)

        # Hard attention: select only the target cube
        attended_cube = (cubes * goal_one_hot.unsqueeze(-1)).sum(dim=1)  # (B, 3)

        ee_to_cube = attended_cube - robot_xyz                  # (B, 3)
        ee_to_goal = goal_pos      - robot_xyz                  # (B, 3)

        return torch.cat([gripper, ee_to_cube, ee_to_goal], dim=1)  # (B, 7)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        x = self._build_input(state)                            # (B, 7)
        flat = self.mlp(x)                                      # (B, chunk_size * action_dim)
        return flat.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(self.forward(state), action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
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
    state_mean: torch.Tensor | None = None,
    state_std:  torch.Tensor | None = None,
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
            p=p,
            state_mean=state_mean,
            state_std=state_std,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
