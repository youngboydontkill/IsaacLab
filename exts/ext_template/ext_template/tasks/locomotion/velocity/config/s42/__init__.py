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
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KuavoS42FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KuavoS42FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Rough-Kuavo-S42-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS42RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42RoughPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Rough-Kuavo-S42-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS42RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42RoughPPORunnerCfg",
    },
)
# for hug wbc
gym.register(
    id="Legged-Isaac-Velocity-Rough-HugWBC-Kuavo-S42-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hugwbc_env_cfg:KuavoS42HugWBCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42RoughHugWBCPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Rough-HugWBC-Kuavo-S42-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hugwbc_env_cfg:KuavoS42HugWBCEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42RoughHugWBCPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-HugWBC-Kuavo-S42-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hugwbc_env_cfg:KuavoS42FlatHugWBCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatHugWBCPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-HugWBC-Kuavo-S42-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hugwbc_env_cfg:KuavoS42FlatHugWBCEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoS42FlatHugWBCPPORunnerCfg",
    },
)

# for Attention 
gym.register(
    id="Legged-Isaac-Attention-Rough-Kuavo-S42-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg:KuavoAttentionRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoAttentionRoughPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Attention-Rough-Kuavo-S42-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg:KuavoAttentionRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KuavoAttentionRoughPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-DreamWaq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KuavoS42FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_dreamwaq_cfg:KuavoS42FlatPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Flat-Kuavo-S42-DreamWaq-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KuavoS42FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_dreamwaq_cfg:KuavoS42FlatPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Rough-Kuavo-S42-DreamWaq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS42RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_dreamwaq_cfg:KuavoS42RoughPPORunnerCfg",
    },
)

gym.register(
    id="Legged-Isaac-Velocity-Rough-Kuavo-S42-DreamWaq-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KuavoS42RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_dreamwaq_cfg:KuavoS42RoughPPORunnerCfg",
    },
)
