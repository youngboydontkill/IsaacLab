"""
Brief : Action Term for HugWBC control, the upper joint can be set with noise_act 
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence
from dataclasses import MISSING

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from collections.abc import Sequence

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers

import ext_template.tasks.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class HugWBCAction(mdp.JointPositionAction):
    """
    Action Term for HugWBC control, the upper joint can be set with noise_act 
    """
    def __init__(self, cfg: HugWBCActionCfg, env: ManagerBasedEnv | ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 获取神奇noise 输入
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.external_act_idx = self.robot.find_joints(cfg.external_act_names)[0]  # return (joint_idx, joint_name)
        self.external_act_dim = len(self.external_act_idx)
        # 获取关节限位和默认位置
        self.dof_pos_limits = torch.zeros(self.external_act_dim, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_pos_limits[:, 0] = self.robot.data.soft_joint_pos_limits[0,self.external_act_idx, 0]
        self.dof_pos_limits[:, 1] = self.robot.data.soft_joint_pos_limits[0,self.external_act_idx, 1]
        self.default_dof_pos = self.robot.data.default_joint_pos[0,self.external_act_idx]  # 默认关节位置
        # 当前关节位置
        self.dof_pos = torch.zeros(self.num_envs, self.external_act_dim, dtype=torch.float, device=self.device, requires_grad=False)
        
    def apply_actions(self):
        policy_act = self.processed_actions
        # 获取noise的输入
        self.dof_pos = self.robot.data.joint_pos[:,self.external_act_idx]
        external_act = self._env.command_manager.get_command("hugwbc_cmd")
        self.disturb_actions = external_act[:,:-2]
        self.disturb_rad_curriculum = external_act[:,-2]
        self.disturb_masks = external_act[:,-1].bool()  # only vaild for replace_action = True
        # 根据mask，将policy的action和external action进行合并
        if self.cfg.curriculum_method == 0:
            disturb_action_clip = self._curriculum_disturb_fusion(policy_act)
        elif self.cfg.curriculum_method == 1:
            disturb_action_clip = self._curriculum_disturb_clipping_mean(policy_act)
        elif self.cfg.curriculum_method == 2:
            disturb_action_clip = self._curriculum_disturb_clipping_mean_rad(policy_act)
        # for replace or add noise 
        if (self.cfg.replace_action):
            # 仅在true的时候才有
            policy_act[:, self.external_act_idx] = torch.where(
                self.disturb_masks.view(-1, 1).repeat(1, self.external_act_dim),
                disturb_action_clip,
                policy_act[:,self.external_act_idx]
            )
        else : 
            policy_act[:, self.external_act_idx]= torch.where(
                self.disturb_masks.view(-1, 1).repeat(1, self.external_act_dim),
                policy_act[:,self.external_act_idx] + disturb_action_clip,
                policy_act[:,self.external_act_idx]
            )
        # clip :
        if self.cfg.clip is not None:
            policy_act = torch.clamp(policy_act, min=self._cllip[:,:,0],max=self._cllip[:,:,1])
        # apply 
        self._asset.set_joint_position_target(policy_act,joint_ids=self._joint_ids)

    # for hugwbc apply :
    def _curriculum_disturb_fusion(self, actions):
        disturb_action = torch.clamp(
            self.disturb_actions,
            (- self.cfg.disturb_rad + self.dof_pos - self.default_dof_pos) / self.cfg.action_scale,
            (self.cfg.disturb_rad + self.dof_pos - self.default_dof_pos) / self.cfg.action_scale
        ) # Nosie or traj Target

        fused_disturb_action = self.disturb_rad_curriculum.unsqueeze(-1) * disturb_action +  \
                         (1 - self.disturb_rad_curriculum.unsqueeze(-1)) * actions[:, self.external_act_idx]
        
        return fused_disturb_action

    def _curriculum_disturb_clipping_mean(self, actions):
        # cliping mean with curriculum
        noise_mean = self.disturb_rad_curriculum.unsqueeze(-1) * (self.dof_pos - self.default_dof_pos)+ \
                (1-self.disturb_rad_curriculum.unsqueeze(-1))  * (actions[:,self.external_act_idx] * self.cfg.action_scale)

        disturb_actions = torch.clamp(
            self.disturb_actions,
            (- self.cfg.disturb_rad + noise_mean)/self.cfg.action_scale,
            (self.cfg.disturb_rad + noise_mean)/self.cfg.action_scale
        )
        return disturb_actions
    
    def _curriculum_disturb_clipping_mean_rad(self, actions):
        # clipping mean with curriculum
        noise_mean = self.disturb_rad_curriculum.unsqueeze(-1) * (self.dof_pos - self.default_dof_pos)+ \
                (1-self.disturb_rad_curriculum.unsqueeze(-1))  * (actions[:, self.external_act_idx] * self.cfg.action_scale)
        
        # clipping action rate with curriculum by rad.
        disturb_actions = torch.clamp(
            self.disturb_actions,
            (- self.cfg.disturb_rad * self.disturb_rad_curriculum.unsqueeze(-1) + noise_mean)/self.cfg.action_scale,
            (self.cfg.disturb_rad * self.disturb_rad_curriculum.unsqueeze(-1) + noise_mean)/self.cfg.action_scale
        )
        return disturb_actions

@configclass
class HugWBCActionCfg(mdp.JointPositionActionCfg):
    class_type: type = HugWBCAction 

    asset_name: str = "robot"
    external_act_names : list[str] = ["zarm_.*_joint"]  # 外部输入的action的名称
    replace_action : bool = True  # 是否替换策略得到的action
    disturb_rad: float = 0.2  # 这是什么玩意
    action_scale: float = 0.25
    disturb_in_last_action: bool = False  # 当前只支持false说是
    curriculum_method: int = 2 