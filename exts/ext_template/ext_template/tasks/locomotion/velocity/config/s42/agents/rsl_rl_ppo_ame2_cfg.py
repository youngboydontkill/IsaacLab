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
    class_name = "Enc2ActorCritic"   #EncVelActorCritic EncActorCritic EncDreamWAQActorCritic
    embedding_dim:int = 64
    attn_dim: int | None = None
    map_channels: int | None = None
    num_heads: int = 8
    local_cnn_channels: tuple[int, ...] = (16, 48)
    pos_embed_dim: int = 16
    local_mlp_hidden: tuple[int, ...] = (96,)
    global_mlp_hidden: tuple[int, ...] = (64,)
    global_dim: int = 64
    query_mlp_hidden: tuple[int, ...] = (96,)
    use_batch_norm: bool = True
    pos_from_map: bool = True
    remove_xy_channels: bool = True
    props_embed_dim: int = 64
    actor_props_encoder_hidden: list[int] = [256, 128]
    critic_props_encoder_hidden: list[int] | None = None
    use2Encoder: bool = False
    load_mask:int = 7+8
    # 这两个取代原有的empirical_normalization
    actor_obs_normalization=True 
    critic_obs_normalization=True
    output_attention=False  # policy是否输出attention,这里主要用于可视化
    velocity_estimation_enabled: bool = False  # 兼容 PPO_AME2 里传入的开关
    use_CENet: bool = False  # 兼容 PPO_AME2 里传入的开关

@configclass
class RslRlPpoEnc2AlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "PPO_AME2"
    # PPO_AME2 所需超参数（与 PPO.__init__ 对齐）
    normalize_advantage_per_mini_batch: bool = False
    # AMP 开关（仅对 PPO_AME2 有效）
    amp_enabled: bool = False
    # 速度估计
    velocity_estimation_enabled: bool = False
    velocity_loss_coef: float = 0.0
    # RND
    rnd_cfg: dict | None = None
    # Symmetry
    symmetry_cfg: dict | None = None
    # Multi-GPU
    multi_gpu_cfg: dict | None = None
    # AMP
    amp_grad_pen_coef: float = 1.0
    amp_grad_pen_lambda: float = 10.0
    amp_loss_coef: float = 0.0
    amp_replay_buffer_size: int = 100000
    min_std: float | None = None
    # Distillation (Enc2ActorCritic)
    distill_loss_coef: float = 0.0
    distill_lr: float = 1e-4

@configclass 
class KuavoAttention2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 80000
    save_interval = 50
    experiment_name = "Kuavo/s42/atten2"
    empirical_normalization = True
    # for rsl rl 3.1
    obs_groups = {
        "policy": ["command","policy"],
        "critic": ["command", "privileged"],
        "perception": ["perception"]
    }

    policy = RslRlPpoEncActorCriticCfg(
            init_noise_std=1.0,
            noise_std_type='log',  # 不加这个会导致std<0无法采样
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            embedding_dim=96,
            map_channels=3,
            num_heads=8,
            local_cnn_channels=(16, 48),
            pos_embed_dim=16,
            local_mlp_hidden=(96,),
            global_mlp_hidden=(64,),
            global_dim=64,
            query_mlp_hidden=(96,),
            use_batch_norm=True,
            pos_from_map=True,
            remove_xy_channels=True,
            props_embed_dim=64,
            actor_props_encoder_hidden=[256, 128],
            critic_props_encoder_hidden=None,
            use2Encoder=False,
                # 31(15+16):velocity 64:critic reconstruction
                load_mask=7+8+32, 
        ) 
    algorithm = RslRlPpoEnc2AlgorithmCfg(
        value_loss_coef=1.0,
        # velocity_loss_coef=1.0,
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
        max_grad_norm=1.0,

    )
    def __post_init__(self):
        super().__post_init__()

        # 默认是没有对称性增强的，需要手动在这里设置以方便支持不同policy的obs 
        # step 1 : 设置好history length和观测的key
        # 84 * 5 = 420
        self.history_len = 1 
        # TODO : 这里的数据增强应该需要重新设计支持tensordict
        self.policy_obs_keys = ["base_lin_vel","base_ang_vel","gravity","joint_pos","joint_vel","action"]
        self.privileged_obs_keys = ["base_lin_vel","base_ang_vel","gravity","joint_pos","joint_vel","action"]  # "feet_contact_force","feet_heights"
        self.command_obs_keys = ["cmd"]
        SymmetryAug.clear()
        SymmetryAug.register_obs("policy",self.policy_obs_keys,self.history_len)
        SymmetryAug.register_obs("privileged",self.privileged_obs_keys,self.history_len)
        SymmetryAug.register_obs("command",self.command_obs_keys,self.history_len)
        SymmetryAug.register_obs_high_dim("perception",mirror_scan_height)

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, 
            use_mirror_loss=True,
            mirror_loss_coeff=2.0, 
            # data_augmentation_func=SymmetryAug.data_augmentation_dict  # 这么写有点问题,他会把整个SymmetryAug.xxx识别为一个callable,但是实际只有后面是
            data_augmentation_func=data_augmentation_dict
        )


@configclass 
class KuavoAttention2RoughPPORunnerPlayCfg(KuavoAttention2RoughPPORunnerCfg):
    obs_groups = {
        "policy": ["command","policy"],
        "critic": ["command", "privileged"],
        "perception": ["perception"]
    }
    policy = RslRlPpoEncActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        embedding_dim=96,
        map_channels=3,
        num_heads=8,
        local_cnn_channels=(16, 48),
        pos_embed_dim=16,
        local_mlp_hidden=(96,),
        global_mlp_hidden=(64,),
        global_dim=64,
        query_mlp_hidden=(96,),
        use_batch_norm=True,
        pos_from_map=True,
        remove_xy_channels=True,
        props_embed_dim=64,
        actor_props_encoder_hidden=[256, 128],
        critic_props_encoder_hidden=None,
        use2Encoder=False,
        load_mask=15+32,
        output_attention=True,
        velocity_estimation_enabled = True,
        use_CENet = False
        )
    def __post_init__(self):
        super().__post_init__()
