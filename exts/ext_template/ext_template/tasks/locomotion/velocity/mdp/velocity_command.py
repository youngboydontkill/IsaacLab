from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from collections.abc import Sequence

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers

import ext_template.tasks.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class UniformSteppingVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: UniformSteppingVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformSteppingVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # -- add command stepping to vel_command_b[:, 3], 1.0 for stepping, 0.0 for standing
        self.vel_command_b = torch.zeros(self.num_envs, 4, device=self.device)
        self.is_stepping_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 4)."""
        return self.vel_command_b
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Added command stepping to vel_command_b[:, 3], 1.0 for stepping, 0.0 for standing"""
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        self.is_stepping_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_stepping_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        non_standing_env_ids = (~self.is_standing_env).nonzero(as_tuple=False).flatten()
        self.vel_command_b[:, 3] = self.is_stepping_env.float()
        # if standing, set velocity to zero, set stepping according to is_stepping_env flag.
        self.vel_command_b[standing_env_ids, :3] = 0.0
        # else non standing,
        self.vel_command_b[non_standing_env_ids, 3] = 1.0


@configclass
class UniformSteppingVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformSteppingVelocityCommand

    rel_stepping_envs: float = 0.0
    """The sampled probability of environments that should be stepping. Defaults to 0.0."""


