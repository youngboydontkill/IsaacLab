# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KuavoS42FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KuavoS42FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Rough-Kuavo-S42-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS42RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42RoughPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Rough-Kuavo-S42-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fough_env_cfg:KuavoS42RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42RoughPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-DreamWaq-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KuavoS42FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_dreamwaq_cfg:KuavoS42FlatPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-DreamWaq-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KuavoS42FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_dreamwaq_cfg:KuavoS42FlatPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Rough-Kuavo-S42-DreamWaq-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS42RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_dreamwaq_cfg:KuavoS42RoughPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Rough-Kuavo-S42-DreamWaq-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS42RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_dreamwaq_cfg:KuavoS42RoughPPORunnerCfg",
    },
)
