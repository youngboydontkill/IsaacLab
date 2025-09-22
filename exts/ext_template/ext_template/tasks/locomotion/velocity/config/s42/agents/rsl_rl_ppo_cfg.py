from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

import torch
import numpy as np
from .symmetry_aug import SymmetryAug2D,SymmetryAug,mirror_scan_height

def data_augmentation_dict_2d(env, obs, actions):
    return SymmetryAug2D.data_augmentation_dict(env, obs, actions)

def data_augmentation_dict(env, obs, actions):
    return SymmetryAug.data_augmentation_dict(env, obs, actions)
@configclass
class KuavoS42RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 50
    experiment_name = "Kuavo/s42/rough"
    empirical_normalization = True
    # for rsl rl 3.0 
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "privileged"],
    }
    policy = RslRlPpoActorCriticCfg(
        # class_name="ActorCriticRecurrent",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0
    )
    
    def __post_init__(self):
        super().__post_init__()
        # 默认是没有对称性增强的，需要手动在这里设置以方便支持不同policy的obs 
        # step 1 : 设置好history length和观测的key
        self.history_len = 5 
        self.policy_obs_keys = ["base_ang_vel","gravity","cmd","joint_pos","joint_vel","action"]
        self.critic_obs_keys = ["base_ang_vel","gravity","cmd","joint_pos","joint_vel","action",
            "base_lin_vel","height_scan","joint_torques","joint_accs","feet_lin_vel","feet_contact_force",
            "base_mass_rel","rigid_body_material","base_com","action_delay","push_force","push_torque",
            "feet_heights","feet_air_times"]
        # step 2 : 设置好对称性增强的规则
        SymmetryAug2D.clear()
        SymmetryAug2D.register_policy_obs(self.policy_obs_keys,self.history_len)
        SymmetryAug2D.register_critic_obs(self.critic_obs_keys,self.history_len)

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, 
            use_mirror_loss=True,
            mirror_loss_coeff=1.0, 
            data_augmentation_func=data_augmentation_dict_2d,
        )
@configclass
class KuavoS42FlatPPORunnerCfg(KuavoS42RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "Kuavo/s42/flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]

        # 默认是没有对称性增强的，需要手动在这里设置以方便支持不同policy的obs 
        # step 1 : 设置好history length和观测的key
        self.history_len = 5 
        self.policy_obs_keys = ["base_ang_vel","gravity","cmd","joint_pos","joint_vel","action"]
        self.critic_obs_keys = ["base_ang_vel","gravity","cmd","joint_pos","joint_vel","action",
            "base_lin_vel","joint_torques","joint_accs","feet_lin_vel","feet_contact_force",
            "base_mass_rel","rigid_body_material","base_com","action_delay","push_force","push_torque",
            "feet_heights","feet_air_times"]
        # step 2 : 设置好对称性增强的规则
        SymmetryAug2D.clear()
        SymmetryAug2D.register_policy_obs(self.policy_obs_keys,self.history_len)
        SymmetryAug2D.register_critic_obs(self.critic_obs_keys,self.history_len)

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, 
            use_mirror_loss=True,
            mirror_loss_coeff=1.0, 
            data_augmentation_func=data_augmentation_dict_2d,
        )
    
@configclass
class KuavoS42RoughHugWBCPPORunnerCfg(KuavoS42RoughPPORunnerCfg):
    # for rsl rl 3.1
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "privileged"],
    }
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "Kuavo/s42/rough/hugwbc"

        # 默认是没有对称性增强的，需要手动在这里设置以方便支持不同policy的obs 
        # step 1 : 设置好history length和观测的key
        self.history_len = 5 
        self.policy_obs_keys = ["base_ang_vel","gravity","cmd_vel","hugwbc_cmd","joint_pos","joint_vel","action"]
        self.critic_obs_keys = ["base_ang_vel","gravity","cmd_vel","hugwbc_cmd","joint_pos","joint_vel","action",
            "base_lin_vel","height_scan","joint_torques","joint_accs","feet_lin_vel","feet_contact_force",
            "base_mass_rel","rigid_body_material","base_com","action_delay","push_force","push_torque",
            "feet_heights","feet_air_times"]
        # step 2 : 设置好对称性增强的规则
        SymmetryAug2D.clear()
        SymmetryAug2D.register_policy_obs(self.policy_obs_keys,self.history_len)
        SymmetryAug2D.register_critic_obs(self.critic_obs_keys,self.history_len)

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, 
            use_mirror_loss=True,
            mirror_loss_coeff=1.0, 
            data_augmentation_func=data_augmentation_dict_2d,
        )

@configclass
class KuavoS42FlatHugWBCPPORunnerCfg(KuavoS42RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "Kuavo/s42/flat/hugwbc"

        # 默认是没有对称性增强的，需要手动在这里设置以方便支持不同policy的obs 
        # step 1 : 设置好history length和观测的key
        self.history_len = 5 
        self.policy_obs_keys = ["base_ang_vel","gravity","cmd_vel","hugwbc_cmd","joint_pos","joint_vel","action"]
        self.critic_obs_keys = ["base_ang_vel","gravity","cmd_vel","hugwbc_cmd","joint_pos","joint_vel","action",
            "base_lin_vel","joint_torques","joint_accs","feet_lin_vel","feet_contact_force",
            "base_mass_rel","rigid_body_material","base_com","action_delay","push_force","push_torque",
            "feet_heights","feet_air_times"]
        # step 2 : 设置好对称性增强的规则
        SymmetryAug2D.clear()
        SymmetryAug2D.register_policy_obs(self.policy_obs_keys,self.history_len)
        SymmetryAug2D.register_critic_obs(self.critic_obs_keys,self.history_len)

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, 
            use_mirror_loss=True,
            mirror_loss_coeff=1.0, 
            # data_augmentation_func="SymmetryAug2D:data_augmentation_dict",
            data_augmentation_func=data_augmentation_dict_2d,
        )

@configclass
class RslRlPpoEncActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "EncActorCritic"
    embedding_dim:int = 64
    load_mask:int = 7+8
    # 这两个取代原有的empirical_normalization
    actor_obs_normalization=True 
    critic_obs_normalization=True
    output_attention=False  # policy是否输出attention,这里主要用于可视化
@configclass 
class KuavoAttentionRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 50
    experiment_name = "Kuavo/s42/rough/atten"
    empirical_normalization = True
    # for rsl rl 3.1
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "privileged"],
        "perception": ["perception"]
    }

    policy = RslRlPpoEncActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            embedding_dim=64,
            load_mask=15 
        )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0
    )
    def __post_init__(self):
        super().__post_init__()

        # 默认是没有对称性增强的，需要手动在这里设置以方便支持不同policy的obs 
        # step 1 : 设置好history length和观测的key
        self.history_len = 1 
        
        self.policy_obs_keys = ["base_ang_vel","gravity","cmd_vel","joint_pos","joint_vel","action"]
        self.privileged_obs_keys = ["base_lin_vel","joint_torques","joint_accs","feet_lin_vel","feet_contact_force",
            "base_mass_rel","rigid_body_material","base_com","action_delay","push_force","push_torque",
            "feet_heights","feet_air_times"]
        SymmetryAug.clear()
        SymmetryAug.register_obs("policy",self.policy_obs_keys)
        SymmetryAug.register_obs("privileged",self.privileged_obs_keys)
        SymmetryAug.register_obs_high_dim("perception",mirror_scan_height)

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, 
            use_mirror_loss=True,
            mirror_loss_coeff=1.0, 
            # data_augmentation_func=SymmetryAug.data_augmentation_dict  # 这么写有点问题,他会把整个SymmetryAug.xxx识别为一个callable,但是实际只有后面是
            data_augmentation_func=data_augmentation_dict
        )

@configclass 
class KuavoAttentionRoughPPORunnerPlayCfg(KuavoAttentionRoughPPORunnerCfg):
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "privileged"],
        "perception": ["perception"]
    }
    policy = RslRlPpoEncActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            embedding_dim=64,
            load_mask=15,
            output_attention=True,
        )
    def __post_init__(self):
        super().__post_init__()