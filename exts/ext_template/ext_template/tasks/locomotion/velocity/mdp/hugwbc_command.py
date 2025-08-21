"""
Brief : Command Term for HugWBC upper joint control 
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence
from dataclasses import MISSING

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


class HugWBCCommand(mdp.CommandTerm):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: HugWBCCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: HugWBCCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # 根据配置的关节名称获取关节数目
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.external_act_idx = self.robot.find_joints(cfg.arm_names)[0]  # return (joint_idx, joint_name)
        self.external_act_dim = len(self.external_act_idx)
        # 获取关节限位和默认位置
        self.dof_pos_limits = torch.zeros(self.external_act_dim, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_pos_limits[:, 0] = self.robot.data.soft_joint_pos_limits[0,self.external_act_idx, 0]
        self.dof_pos_limits[:, 1] = self.robot.data.soft_joint_pos_limits[0,self.external_act_idx, 1]
        self.default_dof_pos = self.robot.data.default_joint_pos[0,self.external_act_idx]  # 默认关节位置
        # 当前关节位置
        self.dof_pos = torch.zeros(self.num_envs, self.external_act_dim, dtype=torch.float, device=self.device, requires_grad=False)
        # hugwbc mask :
        self.update_step = 0  # for command update 
        self.disturb_actions = torch.zeros(self.num_envs, self.external_act_dim, dtype=torch.float, device=self.device, requires_grad=False)
        self.disturb_masks = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.disturb_isnoise = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.interrupt_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.disturb_noise_scale = torch.tensor(cfg.noise_scale).to(self.device).unsqueeze(0) 
        self.disturb_noise_lowerbound = torch.tensor(cfg.noise_lowerbound).to(self.device).unsqueeze(0) #+ 0.15
        if cfg.disturb_rad_curriculum:
            self.disturb_rad_curriculum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        else:
            self.disturb_rad_curriculum = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # man!
        self.noise_disturb_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.noise_env_nums = int(self.num_envs * self.cfg.noise_curriculum_ratio)
        self.noise_disturb_mode[:self.noise_env_nums] = True
        # commands :
        # distributed_actions,rad_curriculum,mask 
        self.commands = torch.zeros(self.num_envs, self.external_act_dim+2, device=self.device, requires_grad=False)
        # else:
        # self.commands = torch.zeros(self.num_envs, self.external_act_dim, device=self.device, requires_grad=False)
        


    def __str__(self) -> str:
        msg = "HugWBCCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg


    @property
    def command(self) -> torch.Tensor:
        return self.commands

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass 

    def _debug_vis_callback(self, event):
        pass 

    def _resample_command(self, env_ids: Sequence[int]):
        update_disturb_mask = self.noise_disturb_mode[env_ids]
        update_disturb_envs = env_ids[update_disturb_mask]
        noise_disturb_mask = self.disturb_masks[env_ids]
        noise_disturb_envs = env_ids[noise_disturb_mask]
        if len(update_disturb_mask) > 0 and self.cfg.disturb_rad_curriculum: 
            self._update_disturb_curriculum_grid(update_disturb_envs, noise_disturb_envs)

    def _update_command(self):
        # 往里面扔噪声,更新命令
        self.update_step += 1
        self.dof_pos = self.robot.data.joint_pos[:,self.external_act_idx]
        if (self.update_step % self.cfg.noise_update_step == 0):
            self._resample_noise()
            self.commands[:,:-2] = self.disturb_actions
            self.commands[:,-2] = self.disturb_rad_curriculum
        self.update_step %= self.cfg.noise_update_step



    def _update_metrics(self):
        pass 

    # hugwbc impl :
    def _update_disturb_curriculum_grid(self, env_ids, noise_env_ids):
        if len(env_ids)==0: return
        # 课程相关,已经移动到curriculums中了

        # resample all noise envs disturb
        self.disturb_masks[env_ids] = (torch.rand(len(env_ids))<=0.5).to(self.device) # Reset with half with disturb.
        is_noise = torch.rand(len(env_ids)) <= self.cfg.noise_ratio
        self.disturb_isnoise[env_ids] = is_noise.to(self.device)
        self.disturb_actions[env_ids] = self.dof_pos[env_ids] - self.default_dof_pos
        if self.cfg.replace_action:
            self.interrupt_mask[env_ids] = self.disturb_masks[env_ids]
        else:
            self.interrupt_mask[env_ids] = self.disturb_masks[env_ids] * (~self.disturb_isnoise[env_ids])
        # TODO : 这里是什么东西
        # for key in self.command_sums.keys():
        #     self.command_sums[key][env_ids] = 0.

        # 默认都写入，反正读取是通过ObsTerm进行设置的
        self.commands[env_ids, -1] = self.interrupt_mask[env_ids].float()
    
    def _resample_noise(self):
        if (self.cfg.uniform_noise):
            scale = self.cfg.uniform_scale
            targets = scale * self.disturb_noise_scale * torch.rand((self.num_envs, self.external_act_dim), device=self.device) \
                + self.disturb_noise_lowerbound + self.disturb_noise_scale * (1-scale)/2
        
            # clip disturb for H1 TODO : 这里需要解决一下防止碰撞
            left_env_mask = targets[:, 1] < 0.5
            targets[left_env_mask][:, [2, 3]] = 0
            right_env_maks =  targets[:, 5] > 0.5
            targets[right_env_maks][:, [6, 7]] = 0

            noise_act =  torch.clamp(
                targets - self.default_dof_pos,
                self.dof_pos_limits[:, 0].view(1,-1).repeat(self.num_envs, 1) - self.default_dof_pos,
                self.dof_pos_limits[:, 1].view(1,-1).repeat(self.num_envs, 1) - self.default_dof_pos
            )
            self.disturb_actions = torch.where(
                self.disturb_isnoise.view(-1,1).repeat(1,self.external_act_dim),
                noise_act/self.cfg.action_scale,
                self.disturb_actions
            )
        else:
            mean = torch.zeros(self.external_act_dim, device=self.device)
            std = torch.ones(self.external_act_dim, device=self.device) * self.cfg.disturb_scale

            noise_act = torch.clamp(
                torch.normal(mean, std) + self.dof_pos - self.default_dof_pos,
                self.dof_pos_limits[:, 0].view(1,-1).repeat(self.num_envs, 1) - self.default_dof_pos,
                self.dof_pos_limits[:, 1].view(1,-1).repeat(self.num_envs, 1) - self.default_dof_pos
            )
            self.disturb_actions = torch.where(
                self.disturb_isnoise.view(-1,1).repeat(1,self.external_act_dim),
                noise_act/self.cfg.action_scale,
                self.disturb_actions
            )

@configclass
class HugWBCCommandCfg(CommandTermCfg):
    """Configuration for the HugWBCCommand term."""
    
    class_type: type = HugWBCCommand

    asset_name: str = "robot"

    arm_names: list[str] = ["zarm_.*_joint"]

    # ranges: Ranges = MISSING

    # for hug wbc impl param:
    distribute_scale: float = 2.0  # 高斯分布标准差
    uniform_noise: bool = True  # 使用均匀还是高斯说是
    switch_prob: float = 0.005
    noise_ratio: float = 1.0  # 这是什么万一
    disturb_rad: float = 0.2  # 这是什么玩意
    noise_update_step: int = 30  # 每多少步仿真更新一次噪声
    # tensor([[-2.9060,  1.3352],
    #     [-2.9060,  1.3352],
    #     [-0.2269,  1.9722],
    #     [-1.9722,  0.2269],
    #     [-1.4137,  1.4137],
    #     [-1.4137,  1.4137],
    #     [-2.4871, -0.1309],
    #     [-2.4871, -0.1309],
    #     [-1.4137,  1.4137],
    #     [-1.4137,  1.4137],
    #     [-1.2086,  0.5978],
    #     [-0.5978,  1.2086],
    #     [-0.6283,  0.6283],
    #     [-0.6283,  0.6283]], device='cuda:0')
    noise_scale: list[float] = [
            5.2, 
            5.2,
            2.1, 
            2.1,
            2.8,
            2.8,
            4.8,
            4.8,
            2.8,
            2.8,
            1.6,
            1.6,
            1.2,
            1.2
        ] # Uniform Distribution Noise for each joint.
    noise_lowerbound: list[float] = [
            -2.6,
            -2.6,
            -0.2,
            -1.9,
            -1.4,
            -1.4,
            -2.4,
            -2.4,
            -1.4,
            -1.4,
            -1.1,
            -0.5,
            -0.6,
            -0.6
        ]
    disturb_rad_curriculum: bool = True  # 是否使用课程学习
    noise_curriculum_ratio: float = 0.5  # 课程学习比例
    replace_action = True # 是否替换动作
    action_scale: float = 0.25
    uniform_scale: float = 1.0  # 均匀采样缩放系数
    max_curriculum: float = 1.0 