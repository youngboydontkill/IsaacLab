"""
Brief : 由于HugWBC使用外部的输入引入的观测，因此需要将观测的获取方式单独写出来
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def hugwbc_external_act_obs(env: ManagerBasedRLEnv, command_name: str="hugwbc_cmd"):
    raw_hugwbc_cmd = env.command_manager.get_command(command_name)
    # 这里只取最后一列作为输入返回
    return raw_hugwbc_cmd[:, -1:].float()
