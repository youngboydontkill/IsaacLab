# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaacsim.core.utils.types import ArticulationActions

from isaaclab.actuators import ImplicitActuator
from isaaclab.actuators import DelayedPDActuator
from isaaclab.utils import DelayBuffer

from .ankle_s42 import joint_to_motor_position, get_joint_dumping_torque

if TYPE_CHECKING:
    from .actuator_cfg import DelayedImplicitActuatorCfg
    from .actuator_cfg import DelayedPDActuatorCfg_S42,LejuDelayedPDActuatorCfg


class DelayedImplicitActuator(ImplicitActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedImplicitActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(
            cfg.max_delay, self._num_envs, device=self._device
        )
        self.velocities_delay_buffer = DelayBuffer(
            cfg.max_delay, self._num_envs, device=self._device
        )
        self.efforts_delay_buffer = DelayBuffer(
            cfg.max_delay, self._num_envs, device=self._device
        )
        # all of the envs
        self._ALL_INDICES = torch.arange(
            self._num_envs, dtype=torch.long, device=self._device
        )

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(
            control_action.joint_positions
        )
        control_action.joint_velocities = self.velocities_delay_buffer.compute(
            control_action.joint_velocities
        )
        control_action.joint_efforts = self.efforts_delay_buffer.compute(
            control_action.joint_efforts
        )
        # compte actuator model
        return super().compute(control_action, joint_pos, joint_vel)


class DelayedPDActuator_S42(DelayedPDActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedPDActuatorCfg_S42
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedPDActuatorCfg_S42, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.friction_static = self._parse_joint_parameter(self.cfg.friction_static, 0.)
        self.activation_vel = self._parse_joint_parameter(self.cfg.activation_vel, torch.inf)
        self.friction_dynamic = self._parse_joint_parameter(self.cfg.friction_dynamic, 0.)

        self.gym2lab = [0, 6, 12, 19, 1, 7, 13, 20, 2, 8, 14, 21, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25]
        self.lab2gym = [0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 24, 3, 7, 11, 15, 19, 23, 25]
        self.dof_pos_illegal = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self.dof_pos_illegal[env_ids] = 0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        # compute errors
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel

        motor_pos = joint_to_motor_position(joint_pos[:, self.lab2gym])[:, self.gym2lab]
        # check
        self.dof_pos_illegal |= torch.isnan(motor_pos).any(dim=-1)
        motor_pos[self.dof_pos_illegal] = 0

        damping_torque = - get_joint_dumping_torque(joint_pos[:, self.lab2gym], motor_pos[:, self.lab2gym], self.damping[:, self.lab2gym], joint_vel[:, self.lab2gym])[:, self.gym2lab]

        # check
        self.dof_pos_illegal |= torch.isnan(damping_torque).any(dim=-1)
        damping_torque[self.dof_pos_illegal] = 0

        # calculate the desired joint torques
        # CSP
        self.computed_effort = self.stiffness * error_pos + damping_torque + control_action.joint_efforts \
            - (self.friction_static * torch.tanh(joint_vel / self.activation_vel) - self.friction_dynamic * joint_vel)
        # CST
        # self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts \
        #     - (self.friction_static * torch.tanh(joint_vel / self.activation_vel) + self.friction_dynamic * joint_vel)
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


class LejuDelayedPDActuator(DelayedPDActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: LejuDelayedPDActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: LejuDelayedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.friction_static = self._parse_joint_parameter(self.cfg.friction_static, 0.)
        self.friction_activation_vel = self._parse_joint_parameter(self.cfg.activation_vel, torch.inf)
        self.friction_dynamic = self._parse_joint_parameter(self.cfg.friction_dynamic, 0.)
        # prepare joint vel buffer for max effort computation
        self._joint_vel = torch.zeros_like(self.computed_effort)
        # create buffer for zeros effort
        self._zeros_effort = torch.zeros_like(self.computed_effort)
        # build saturation effort limit tensor
        self.saturation_effort = self._parse_joint_parameter(self.cfg.effort_limit, 0.)
        self.effort_weaken_velocity_limit = self._parse_joint_parameter(self.cfg.effort_weaken_velocity_limit, 0.)
    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self._joint_vel[env_ids] = 0.0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current joint vel
        self._joint_vel[:] = joint_vel
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        # compute errors
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel

        # calculate the desired joint torques with friction compensation
        self.computed_effort = (
            self.stiffness * error_pos 
            + self.damping * error_vel 
            + control_action.joint_efforts
            - (self.friction_static * torch.tanh(joint_vel / self.friction_activation_vel) 
            + self.friction_dynamic * joint_vel)
        )
        
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        
        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        k = -self.saturation_effort / (self.velocity_limit - self.velocity_limit/2)
        effort_limit = torch.clip(k * (self._joint_vel.abs() - self.velocity_limit/2) + self.saturation_effort,min=0.0)
        # check if the joint velocity is less than the max speed at full torque
        max_effort = torch.where(
            self._joint_vel.abs() < self.effort_weaken_velocity_limit, self.saturation_effort, effort_limit)
        # check if the effort is the same direction as the joint velocity
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(same_direction, max_effort, self.saturation_effort)

        return torch.clip(effort, min=-max_effort, max=max_effort)
