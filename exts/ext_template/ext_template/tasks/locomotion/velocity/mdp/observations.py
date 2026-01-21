# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse,quat_apply_yaw, quat_inv 

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class rigid_body_masses(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(
                self.asset.num_bodies, dtype=torch.int, device="cpu"
            )
        else:
            self.body_ids = torch.tensor(
                self.asset_cfg.body_ids, dtype=torch.int, device="cpu"
            )

        self.sum_mass = torch.sum(
            self.asset.root_physx_view.get_masses()[:, self.body_ids].to(env.device),
            dim=-1,
        ).unsqueeze(-1)
        self.count = 0

    def __call__(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.count < 5:
            self.count += 1
            self.sum_mass = torch.sum(
                self.asset.root_physx_view.get_masses()[:, self.body_ids].to(
                    env.device
                ),
                dim=-1,
            ).unsqueeze(-1)
        return self.sum_mass


class rigid_body_material(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(
                self.asset.num_bodies, dtype=torch.int, device="cpu"
            )
        else:
            self.body_ids = torch.tensor(
                self.asset_cfg.body_ids, dtype=torch.int, device="cpu"
            )

        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(
            None
        ):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
        self.idxs = []
        for body_id in self.body_ids:
            idx = sum(self.num_shapes_per_body[:body_id])
            self.idxs.append(idx)

        materials = self.asset.root_physx_view.get_material_properties()
        self.materials = (
            materials[:, self.idxs].reshape(env.num_envs, -1).to(env.device)
        )

        self.count = 0

    def __call__(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.count < 5:
            self.count += 1
            materials = self.asset.root_physx_view.get_material_properties()
            self.materials = (
                materials[:, self.idxs].reshape(env.num_envs, -1).to(env.device)
            )
        return self.materials


class base_com(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(
                self.asset.num_bodies, dtype=torch.int, device="cpu"
            )
        else:
            self.body_ids = torch.tensor(
                self.asset_cfg.body_ids, dtype=torch.int, device="cpu"
            )

        self.coms = (
            self.asset.root_physx_view.get_coms()[:, self.body_ids, :3]
            .to(env.device)
            .squeeze(1)
        )
        self.count = 0

    def __call__(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.count < 5:
            self.count += 1
            self.coms = (
                self.asset.root_physx_view.get_coms()[:, self.body_ids, :3]
                .to(env.device)
                .squeeze(1)
            )
        return self.coms


def contact_information(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    data = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    contact_information = torch.sum(torch.square(data), dim=-1) > 1

    return contact_information.float()


def action_delay(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuators_names: str = "base_legs",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return (
        asset.actuators[actuators_names]
        .positions_delay_buffer.time_lags.float()
        .to(env.device)
        .unsqueeze(1)
    )


def joint_torques(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def joint_accs(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # print(asset.data.joint_names)
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return contact_force.flatten(1, 2)


def feet_lin_vel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    return body_lin_vel_w.flatten(1)


def push_force(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    external_force_b = asset._external_force_b[:, asset_cfg.body_ids, :]
    return external_force_b.flatten(1)


def push_torque(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    external_torque_b = asset._external_torque_b[:, asset_cfg.body_ids, :]
    return external_torque_b.flatten(1)


def feet_heights_bipeds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg1: SceneEntityCfg | None = None,
    sensor_cfg2: SceneEntityCfg | None = None,
) -> torch.Tensor:
    foot_heights = torch.stack(
        [
            env.scene[sensor_cfg.name].data.pos_w[:, 2]
            - env.scene[sensor_cfg.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor_cfg in [sensor_cfg1, sensor_cfg2]
            if sensor_cfg is not None
        ],
        dim=-1,
    )
    foot_heights = torch.nan_to_num(foot_heights, nan=0, posinf=0, neginf=0)
    foot_heights = torch.clamp(foot_heights - 0.02, min=0.0)

    return foot_heights.flatten(1)


def feet_air_time_obs(
        env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    return air_time

# for Perception : 
def map_scan_base(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.
    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    :return : (N, L, W, 3) tensor of height scans 
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # calculate the height scan shape 
    grid_size = sensor.cfg.pattern_cfg.size  # [L,W](m)
    resolution = sensor.cfg.pattern_cfg.resolution  # 
    grid_shape = (int(grid_size[0]/resolution) + 1, int(grid_size[1]/resolution) +1)
    # height scan: height = sensor_height - hit_point_z - offset
    height_scan = sensor.data.pos_w[:, :3].unsqueeze(1) - sensor.data.ray_hits_w[..., :3]
    # shape = [B, L*W, 3]
    L = grid_shape[0]
    W = grid_shape[1]
    B = height_scan.shape[0]
    # convert to base frame
    if sensor.cfg.ray_alignment == "yaw":
        # only yaw orientation is considered and directions are not rotated
        quat_w2b = quat_inv(sensor.data.quat_w)
        height_scan = quat_apply_yaw(quat_w2b.repeat(1, L*W), 
                                      height_scan)
    elif sensor.cfg.ray_alignment == "base":
        height_scan = quat_apply_inverse(sensor.data.quat_w.repeat(1, L*W), 
                                      height_scan)
    # 需要注意,这里的shape是(W,L,3),而不是(L,W,3),因为默认是xy采样,因此会先变化x,
    # 然而先变化x说明先对L进行操作,因此第一个维度是W
    return height_scan.view(B, W, L, 3).permute(0,2,1,3)  # [B,L,W,3]

def fixed_zero_vel(env: ManagerBasedEnv):
    """
    返回固定的零线速度 [0, 0, 0]
    
    参数:
        env: 环境实例（框架会自动传入）
    
    返回:
        torch.Tensor: 形状为 [num_envs, 3] 的零速度张量
    """
    # 获取环境数量，确保输出形状与环境匹配
    num_envs = env.num_envs
    # 返回形状为 [num_envs, 3] 的零张量，与其他观测项格式一致
    return torch.zeros(num_envs, 3, device=env.device)


def end_eff_pos_amp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, end_effector_body_id: int):
    """
    计算末端执行器相对于基础的位置信息，用于AMP任务中的观测项。

    参数:
        env: 环境实例（框架会自动传入）
        asset_cfg: 资产配置，包含机器人信息
        end_effector_body_id: [zarm_l7, zarm_r7, leg_l6,leg_r6_link]中末端执行器的body id
    返回:    
        torch.Tensor: 形状为 [num_envs, 3] 的末端执行器位置张量
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base_pos = asset.data.body_pos_w[:, 0, :]  # 基座位置
    end_eff_pos = asset.data.body_pos_w[:, end_effector_body_id, :]  # 末端执行器位置
    rel_end_eff_pos = end_eff_pos - base_pos  # 相对于基座的位置
    return rel_end_eff_pos
    