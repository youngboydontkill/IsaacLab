from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from ext_template.sensors.volume_points import VolumePoints

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float,
    use_stance_mask: bool = True
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    if (not use_stance_mask):
        # no reward for zero command
        reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    else:
        # no reward for not stepping
        reward *= env.command_manager.get_command(command_name)[:, 3]
    return reward


def feet_air_time_clip(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg,
    threshold_min: float,
    threshold_max: float,
    use_stance_mask: bool = True
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    air_time = (last_air_time - threshold_min) * first_contact
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)
    if (not use_stance_mask):
        # no reward for zero command
        reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) >= 0.1
    else:
        # no reward for not stepping
        # cmd = env.command_manager.get_command(command_name)[:, 3]
        # not_stepping_indices = torch.where(cmd < 0.5)[0]
        # print(f"Indices not stepping: {not_stepping_indices.shape[0]}")
        reward *= env.command_manager.get_command(command_name)[:, 3]
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg,
    use_stance_mask: bool = True
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    if (not use_stance_mask):
        # no reward for zero command
        reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.01
    else :
        # no reward for not stepping
        reward *= env.command_manager.get_command(command_name)[:, 3]
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def volume_points_penetration(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, tolerance: float = 0.0
) -> torch.Tensor:
    """Penalize the penetration of volume points into the environment."""
    volume_sensor: VolumePoints = env.scene.sensors[sensor_cfg.name]
    penetration = volume_sensor.data.penetration_offset  # (N, B_, P_, 3)
    penetration = penetration.flatten(1, 2)  # (N, B_*P_, 3)
    penetration_depth = torch.norm(penetration, dim=-1)  # (N, B_*P_)
    in_obstacle = (penetration_depth > tolerance).float()  # (N, B_*P_)
    points_vel = volume_sensor.data.points_vel_w  # (N, B_, P_, 3)
    points_vel = points_vel.flatten(1, 2)  # (N, B_*P_, 3)
    points_vel_norm = torch.norm(points_vel, dim=-1)  # (N, B_*P_)
    velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth
    return torch.sum(velocity_times_penetration, dim=-1)

def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
    use_stance_mask: bool = False,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    if (not use_stance_mask):
        reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    else :
        # no reward for not stepping
        reward *= env.command_manager.get_command(command_name)[:, 3]
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset : Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str,sensor_cfg: SceneEntityCfg,
                             use_stance_mask: bool = True) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contacts = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contacts, dim=-1).float()
    if (not use_stance_mask):
        cmd = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :3],dim=1)
        reward *= cmd< 0.01
    else:
        # no reward for stepping
        reward *= (env.command_manager.get_command(command_name)[:, 3] < 0.5)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def dof_vel_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(joint_vel**2, dim=1)

def joint_power_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    joint_power = (
        asset.data.applied_torque[:, asset_cfg.joint_ids]
        * asset.data.joint_vel[:, asset_cfg.joint_ids]
    )

    return torch.sum(torch.abs(joint_power), dim=1)


class action_smoothness_l2(ManagerTermBase):
    def __init__(
        self, env: ManagerBasedEnv, cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        super().__init__(env, cfg)
        self.prev_prev_action = None

    def __call__(
        self, env: ManagerBasedEnv, cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.prev_prev_action is None:
            self.prev_prev_action = env.action_manager.prev_action.clone()
        action_smoothness_l2 = torch.sum(
            torch.square(
                env.action_manager.action
                - 2 * env.action_manager.prev_action
                + self.prev_prev_action
            ),
            dim=1,
        )
        self.prev_prev_action = env.action_manager.prev_action.clone()
        return action_smoothness_l2


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        base_height = asset.data.root_pos_w[:, 2] - sensor.data.ray_hits_w[..., 2].mean(
            dim=-1
        )
    else:
        base_height = asset.data.root_link_pos_w[:, 2]
    # Replace NaNs with the base_height
    base_height = torch.nan_to_num(
        base_height, nan=target_height, posinf=target_height, neginf=target_height
    )

    # Compute the L2 squared penalty
    return torch.square(base_height - target_height)


def track_lin_vel_xy_yaw_frame_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_link_quat_w), asset.data.root_com_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]
        ),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # 这里需要投影到惯性系
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]
        - asset.data.root_com_ang_vel_w[:, 2]
    )
    reward = torch.exp(-ang_vel_error / std**2)
    # 同样投影到惯性系
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_default_arm_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    alpha: float = 5.0,
) -> torch.Tensor:
    """奖励手臂关节追踪默认位置"""
    asset: Articulation = env.scene[asset_cfg.name]
    arm_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [num_envs, num_joints]

    sq_dist = torch.sum((arm_joint_pos-0.0)**2, dim=1)
    reward = torch.exp(-alpha * sq_dist)

    return reward

def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, violation_max: float = torch.inf) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    return torch.sum(violation.clip(min=0.0, max=violation_max), dim=1)


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_stance_mask: bool = True,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :3],dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:,:2], dim=1)
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    stance_reward = torch.linalg.norm(diff_angle, dim=1)
    if (not use_stance_mask):
        reward = torch.where(torch.logical_or(cmd > 0.01,body_vel > 0.5),0.0,stance_reward)
    else:
        # no reward for stepping
        is_stepping = env.command_manager.get_command(command_name)[:, 3] > 0.5
        reward = torch.where(torch.logical_or(is_stepping,body_vel > 0.5),0.0,stance_reward)
    # reward = torch.sum(torch.abs(diff_angle), dim=-1)
    # reward *= (
    #     torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1
    # )
    reward *= torch.clamp(-asset.data.projected_gravity_b[:,2],0,0.7) / 0.7
    return reward

def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # 是否踉跄
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 返回接触传感器数据中，在指定身体ID上的净力在x和y方向上的模大于3倍的z方向上的模的索引
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 3 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )

def no_feet_contact(env: ManagerBasedRLEnv, 
                    command_name: str, 
                    sensor_cfg: SceneEntityCfg, 
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    use_stance_mask:bool = True) -> torch.Tensor:
    """
    加入这一项是为了提升在有速度指令情况下的响应速度
    Penalize no feet contact
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :3],dim=1)
    # compute the reward
    # 获取接触传感器的数据，并计算每个接触点的力的大小
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 5.0
    # 计算没有接触的接触点数量
    no_contact = contacts.sum(dim=1) == 0
    # 如果没有接触的接触点数量为0，并且速度指令小于0.5，则奖励为1.0，否则为0.0
    if (not use_stance_mask):
        reward = torch.where(
            torch.logical_and(no_contact, cmd<0.2),
            1.0,
            0.0,
        )
        return reward
    else:
        # no reward for not stepping
        not_stepping = env.command_manager.get_command(command_name)[:, 3] < 0.5
        reward = torch.where(
            torch.logical_and(no_contact, not_stepping),
            1.0,
            0.0,
        )
        return reward

def flat_orientation_l2_lean(
    env: ManagerBasedRLEnv, 
    target_pitch: float = 0.05,  # 前倾目标角度（弧度）
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-flat base orientation with a target pitch angle using L2 squared kernel."""
    # 提取机器人对象
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 计算目标x分量（假设重力向量归一化，且前倾为绕y轴旋转）
    target_x = torch.sin(torch.tensor(target_pitch, device=asset.device))
    
    # 当前重力投影的x/y分量
    current_x = asset.data.projected_gravity_b[:, 0]
    current_y = asset.data.projected_gravity_b[:, 1]
    
    # 计算误差的平方和（x方向偏移target_x，y方向保持为0）
    return (current_x - target_x).pow(2) + current_y.pow(2)

def illegal_dof_pos_barrier(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚串联关节位置，避免进入可行区域外"""
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取两个ankle关节的位置
    ankle_joint_pos = asset.data.joint_pos[:,asset_cfg.joint_ids]
    # 凸可行域定义
    vertices = torch.tensor(
        [
            [0.46,0],
            [0,0.8],
            [-0.5,0.8],
            [-0.87,0.5],
            [-0.87,-0.5],
            [-0.5,-0.8],
            [0,-0.8]
        ], device=ankle_joint_pos.device
    )
    feasible_region = torch.tensor(
        [
            [0.87,0.5,-0.40],
            [-1,0,-0.87],
            [-0.63,-0.78,-0.94],
            [0.87,-0.5,-0.4],
            [0,-1,-0.8],
            [-0.63,0.78,-0.94],
            [0,1,-0.8]
        ],device=ankle_joint_pos.device
    )
    # 构建不等式约束 Ax - b \leq 0 -> -log(b-Ax)
    eps = 0.05 # 约束的松弛系数
    left_side = -feasible_region[:,-1] - torch.matmul(ankle_joint_pos[:,:2], feasible_region[:,:-1].T)
    right_side = -feasible_region[:,-1] - torch.matmul(ankle_joint_pos[:,2:],feasible_region[:,:-1].T)
    # 不知道为什么会有并联解算后为nan的情况，这个应该直接terminate掉的，这里直接忽略掉可行域外的
    left_mask =(left_side <-eps).any(dim=-1)
    right_mask = (right_side <-eps).any(dim=-1)
    outside = torch.logical_or(left_mask, right_mask).sum()
    # if outside > 0:
    #     print(f"Warning: {outside} envs are outside the feasible region, this should be terminated.")
    left_side[left_mask] = 0.0
    right_side[right_mask] = 0.0

    # 使用log构建barrier function(默认是可行的)，并进行截断
    max_penalty = 25.0
    l_penalty = torch.clamp(
        -torch.log(left_side + eps), min=0, max=max_penalty
    )
    r_penalty = torch.clamp(
        -torch.log(right_side + eps), min=0, max=max_penalty
    )

    penalty = (l_penalty + r_penalty).sum(dim=1)
    # lmask = l_penalty.isnan().sum(dim=1) > 0
    # rmask = r_penalty.isnan().sum(dim=1) > 0
    # print(ankle_joint_pos[lmask])
    # print(ankle_joint_pos[rmask])
    # print(l_penalty.isnan().sum(), r_penalty.isnan().sum())
    # 返回惩罚项
    return penalty 
def feet_too_near_humanoid(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.16
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    desired_links = ["leg_l6_link", "leg_r6_link"]
    body_ids, _ = asset.find_bodies(desired_links)
    # eef: RigidObject = env.scene[eef_cfg.name]
    # base_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    # base_link_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    # base_link_pos_w = asset.data.root_pos_w - asset.data.root_pos_w
    base_link_quat_w = asset.data.root_quat_w
    eef_link_pos_w = asset.data.body_pos_w[:, body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    # eef_link_quat_w = eef.data.body_link_quat_w[:, eef_cfg.body_ids, :]
    eef_pos_body = quat_apply_inverse(base_link_quat_w[:, None, :].expand(-1, eef_link_pos_w.shape[1], -1), 
                                      eef_link_pos_w)
    # print("eef_pos_body: ", eef_pos_body[0, :, :])
    distance = eef_pos_body[:, 0, 1] - eef_pos_body[:, 1, 1]
    return (threshold - distance).clamp(min=0)

def fly(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # print("is_contact.shape: ", is_contact.shape)
    return torch.sum(is_contact, dim=-1) < 0.5


def feet_solid_contact(env: ManagerBasedRLEnv, 
                       # contact force sensor
                       sensor_cfg: SceneEntityCfg, 
                       # 2x feet height sensor
                       sensor_cfg1: SceneEntityCfg | None = None,
                       sensor_cfg2: SceneEntityCfg | None = None,
                       # contact force threshold
                       threshold=5.0,
                       # feet height threshold, feet heights that smaller than this is considered as solid contact
                       # on Kuavo s46, feet_height is 0.07m when feet are on the ground.
                       feet_height_threshold=0.13):
    """
    Penalize not solid contact for stair environment.
    When contact force is larger than threshold, penalize corresponding feet height.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # decide is contact
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # print("net_contact_forces.shape: ", net_contact_forces.shape)
    is_contact = torch.norm(net_contact_forces[:, 0, sensor_cfg.body_ids], dim=-1) > threshold
    # print("is_contact.shape: ", is_contact.shape)
    # print("is_contact[0]: ", is_contact[0, :])

    # get feet heights
    foot_heights_raw = torch.stack(
        [
            # expand pos_w[:,2] to (N,1) so it can broadcast with ray_hits_w[...,2] which is (N,M)
            env.scene[sensor_cfg.name].data.pos_w[:, 2].unsqueeze(-1)
            - env.scene[sensor_cfg.name].data.ray_hits_w[..., 2]
            for sensor_cfg in [sensor_cfg1, sensor_cfg2]
            if sensor_cfg is not None
        ],
        dim=1,
    )
    # print("foot_heights_raw.shape: ", foot_heights_raw.shape)
    # print("foot_heights_raw[0]: ", foot_heights_raw[0, :])
    feet_height = foot_heights_raw - feet_height_threshold
    # clip feet_height to [0.0, 0.2]
    feet_height = torch.clamp(feet_height, min=0.0, max=0.2)
    # is_contact: [N,2], feet_height: [N,2,10]
    # 只在 is_contact 为 True 的位置上对最后一维求和 -> [N,2]
    # contact_heights_sum = torch.sum(feet_height * is_contact.unsqueeze(-1), dim=-1)
    # 使用 torch.where 在非接触处填 0，再沿最后一维求和 -> [N,2]
    mask = is_contact.unsqueeze(-1)  # [N,2,1]
    contact_heights_sum = torch.sum(torch.where(mask, feet_height, torch.zeros_like(feet_height)), dim=-1)
    # print("contact_heights_sum: ", contact_heights_sum, "\n")
    return torch.sum(contact_heights_sum)
    # # 判定：当有接触且对应的高度和 > 0 时认为该脚接触不实
    # feet_not_solid_per_foot = is_contact & (contact_heights_sum > 0.0)
    # # 返回每个 env 中不稳的脚数量（可作为惩罚值），类型为 float
    # feet_not_solid = feet_not_solid_per_foot.sum(dim=1).to(feet_height.dtype)
    # return feet_not_solid
