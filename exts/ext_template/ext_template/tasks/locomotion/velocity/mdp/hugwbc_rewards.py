from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv

def joint_power_l2_wbc(
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
    external_act = env.command_manager.get_command("hugwbc_cmd")
    external_act_mask = external_act[:,-1].bool()
    indices = asset.find_joints(["zarm_.*_joint"])[0]
    
    if isinstance(asset_cfg.joint_ids, slice):
        # 生成切片对应的索引列表
        joint_ids_list = list(range(asset.data.applied_torque.shape[1]))[asset_cfg.joint_ids]
    else:
        joint_ids_list = asset_cfg.joint_ids

    # 获取需要排除的关节索引与实际关节索引的交集
    common_indices = list(set(indices) & set(joint_ids_list))
    # 找到交集中每个索引在 joint_ids_list 中的位置（即 joint_torque_square 中的列索引）
    external_act_ids_in_torque = [joint_ids_list.index(idx) for idx in common_indices]
    external_act_ids = torch.tensor(external_act_ids_in_torque, dtype=torch.int32, device=env.device)

    joint_power[external_act_mask][:,external_act_ids] = 0.0 #for wbc 

    return torch.sum(torch.abs(joint_power), dim=1)

def joint_torques_l2_wbc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_torque_square = torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids])
    external_act = env.command_manager.get_command("hugwbc_cmd")
    external_act_mask = external_act[:,-1].bool()
    indices = asset.find_joints(["zarm_.*_joint"])[0]
    if isinstance(asset_cfg.joint_ids, slice):
        # 生成切片对应的索引列表
        joint_ids_list = list(range(asset.data.applied_torque.shape[1]))[asset_cfg.joint_ids]
    else:
        joint_ids_list = asset_cfg.joint_ids

    # 获取需要排除的关节索引与实际关节索引的交集
    common_indices = list(set(indices) & set(joint_ids_list))
    # 找到交集中每个索引在 joint_ids_list 中的位置（即 joint_torque_square 中的列索引）
    external_act_ids_in_torque = [joint_ids_list.index(idx) for idx in common_indices]
    external_act_ids = torch.tensor(external_act_ids_in_torque, dtype=torch.int32, device=env.device)

    joint_torque_square[external_act_mask][:,external_act_ids] = 0.0 #for wbc

    return torch.sum(joint_torque_square, dim=1)

def joint_acc_l2_wbc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc_square = torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids])
    external_act = env.command_manager.get_command("hugwbc_cmd")
    external_act_mask = external_act[:,-1].bool()
    indices = asset.find_joints(["zarm_.*_joint"])[0]
    if isinstance(asset_cfg.joint_ids, slice):
        # 生成切片对应的索引列表
        joint_ids_list = list(range(asset.data.joint_acc.shape[1]))[asset_cfg.joint_ids]
    else:
        joint_ids_list = asset_cfg.joint_ids

    # 获取需要排除的关节索引与实际关节索引的交集
    common_indices = list(set(indices) & set(joint_ids_list))
    # 找到交集中每个索引在 joint_ids_list 中的位置（即 joint_torque_square 中的列索引）
    external_act_ids_in_torque = [joint_ids_list.index(idx) for idx in common_indices]
    external_act_ids = torch.tensor(external_act_ids_in_torque, dtype=torch.int32, device=env.device)

    joint_acc_square[external_act_mask][:,external_act_ids] = 0.0 #for wbc
    return torch.sum(joint_acc_square, dim=1)

def action_rate_l2_wbc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    action_rate_square = torch.square(env.action_manager.action - env.action_manager.prev_action)
    external_act = env.command_manager.get_command("hugwbc_cmd")
    external_act_mask = external_act[:,-1].bool()
    indices = asset.find_joints(["zarm_.*_joint"])[0]

    if isinstance(asset_cfg.joint_ids, slice):
        # 生成切片对应的索引列表
        joint_ids_list = list(range(env.action_manager.action.shape[1]))[asset_cfg.joint_ids]
    else:
        joint_ids_list = asset_cfg.joint_ids

    # 获取需要排除的关节索引与实际关节索引的交集
    common_indices = list(set(indices) & set(joint_ids_list))
    # 找到交集中每个索引在 joint_ids_list 中的位置（即 joint_torque_square 中的列索引）
    external_act_ids_in_torque = [joint_ids_list.index(idx) for idx in common_indices]
    external_act_ids = torch.tensor(external_act_ids_in_torque, dtype=torch.int32, device=env.device)

    action_rate_square[external_act_mask][:,external_act_ids] = 0.0 #for wbc
    return torch.sum(action_rate_square, dim=1)

class action_smoothness_l2_wbc(ManagerTermBase):
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
        asset: Articulation = env.scene[cfg.name]
        external_act = env.command_manager.get_command("hugwbc_cmd")
        external_act_mask = external_act[:,-1].bool()
        indices = asset.find_joints(["zarm_.*_joint"])[0]
        external_act_ids = torch.tensor(indices,dtype=torch.int32, device=env.device) 
        action_smoothness = torch.square(
                env.action_manager.action
                - 2 * env.action_manager.prev_action
                + self.prev_prev_action
            )
        action_smoothness[external_act_mask][:,external_act_ids] = 0.0 #for wbc
        action_smoothness_l2 = torch.sum(action_smoothness,dim=1)
        self.prev_prev_action = env.action_manager.prev_action.clone()
        return action_smoothness_l2

def joint_deviation_arms_wbc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    external_act = env.command_manager.get_command("hugwbc_cmd")
    external_act_mask = external_act[:,-1].bool()
    indices = asset.find_joints(["zarm_.*_joint"])[0]

    if isinstance(asset_cfg.joint_ids, slice):
        # 生成切片对应的索引列表
        joint_ids_list = list(range(env.action_manager.action.shape[1]))[asset_cfg.joint_ids]
    else:
        joint_ids_list = asset_cfg.joint_ids

    # 获取需要排除的关节索引与实际关节索引的交集
    common_indices = list(set(indices) & set(joint_ids_list))
    # 找到交集中每个索引在 joint_ids_list 中的位置（即 joint_torque_square 中的列索引）
    external_act_ids_in_torque = [joint_ids_list.index(idx) for idx in common_indices]
    external_act_ids = torch.tensor(external_act_ids_in_torque, dtype=torch.int32, device=env.device)

    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    angle[external_act_mask][:,external_act_ids] = 0.0 #for wbc
    return torch.sum(torch.abs(angle), dim=1)

def stand_still_without_cmd_wbc(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_stance_mask: bool = True
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :3],dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:,:2], dim=1)
    external_act = env.command_manager.get_command("hugwbc_cmd")
    external_act_mask = external_act[:,-1].bool()
    indices = asset.find_joints(["zarm_.*_joint"])[0]
    
    if isinstance(asset_cfg.joint_ids, slice):
        # 生成切片对应的索引列表
        joint_ids_list = list(range(env.action_manager.action.shape[1]))[asset_cfg.joint_ids]
    else:
        joint_ids_list = asset_cfg.joint_ids

    # 获取需要排除的关节索引与实际关节索引的交集
    common_indices = list(set(indices) & set(joint_ids_list))
    # 找到交集中每个索引在 joint_ids_list 中的位置（即 joint_torque_square 中的列索引）
    external_act_ids_in_torque = [joint_ids_list.index(idx) for idx in common_indices]
    external_act_ids = torch.tensor(external_act_ids_in_torque, dtype=torch.int32, device=env.device)

    # 对手部的关节索引，使用external_act_mask进行赋值，为true对应的diff设置为0
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    diff_angle[external_act_mask][:,external_act_ids] = 0.0 #for wbc 
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