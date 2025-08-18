from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

import torch
import numpy as np

# define symmetry rules for single observation
act_mirror_indices = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24]
act_mirror_signs = [-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1]

# define obs&act length and check the symmetry rules, define history length
POLICY_MIRROR_INDICES = []
POLICY_MIRROR_SIGNS = []
CRITIC_MIRROR_INDICES = []
CRITIC_MIRROR_SIGNS = []
ACT_MIRROR_INDICES = act_mirror_indices
ACT_MIRROR_SIGNS = act_mirror_signs

# expand policy obs symmetry rules w.r.t history length
singleObs = {
            "base_ang_vel":[[0, 1, 2],[-1,1,-1]],
            "gravity":[[0,1,2],[1,-1,1]],
            "cmd":[[0,1,2,3],[1, -1, -1, 1]],
            "joint_pos":[[1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10,
                            13, 12, 15, 14, 17,16, 19, 18, 21, 20, 23, 22, 25, 24],
                        [-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1]],
            "joint_vel":[[1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10,
                            13, 12, 15, 14, 17,16, 19, 18, 21, 20, 23, 22, 25, 24],
                        [-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1]],
            "action":[[1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10,
                            13, 12, 15, 14, 17,16, 19, 18, 21, 20, 23, 22, 25, 24],
                        [-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1]]
        }

singleCriticObs = {
            "base_ang_vel":[[0, 1, 2],[-1,1,-1]],
            "gravity":[[0,1,2],[1,-1,1]],
            "cmd":[[0,1,2,3],[1, -1, -1, 1]],
            "joint_pos":[[1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10,
                            13, 12, 15, 14, 17,16, 19, 18, 21, 20, 23, 22, 25, 24],
                        [-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1]],
            "joint_vel":[[1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10,
                            13, 12, 15, 14, 17,16, 19, 18, 21, 20, 23, 22, 25, 24],
                        [-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1]],
            "action":[[1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10,
                            13, 12, 15, 14, 17,16, 19, 18, 21, 20, 23, 22, 25, 24],
                        [-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1]],
            "base_lin_vel": [[0, 1, 2],[1, -1, 1]],
            # 高程图，1.6m x 1.0m，分辨率0.1m，共17x11个点，排列顺序为xy，所以关于y对称就是每隔17个点为一列，把这11列倒序排列即可。符号不变。
            "height_scan": [[170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
                             153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
                             136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
                             119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
                             102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                             85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
                             68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                             51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                             34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                            [1 for i in range(187)]],
            "joint_torques": [[1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10,
                            13, 12, 15, 14, 17,16, 19, 18, 21, 20, 23, 22, 25, 24],
                        [-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1]],
            "joint_accs": [[1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10,
                            13, 12, 15, 14, 17,16, 19, 18, 21, 20, 23, 22, 25, 24],
                        [-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1]],
            "feet_lin_vel": [[3, 4, 5, 0, 1, 2],[1, -1, 1, 1, -1, 1]],
            "feet_contact_force": [[3, 4, 5, 0, 1, 2],[1, -1, 1, 1, -1, 1]],
            "base_mass_rel": [[0,],[1,]],
            "rigid_body_material": [[3, 4, 5, 0, 1, 2],[1, 1, 1, 1, 1, 1]],
            "base_com": [[0,1,2],[1,-1,1]],
            "action_delay": [[0,],[1,]],
            "push_force": [[0,1,2],[1,-1,1]],
            "push_torque": [[0, 1, 2],[-1,1,-1]],
            "feet_heights": [[1,0],[1,1]],
            "feet_air_times": [[1,0],[1,1]],
}

def mirror_tensors(tensor, mirror_indices, mirror_signs):
    mirrored_tensor = tensor[..., mirror_indices]
    mirror_signs_tensor = torch.tensor(mirror_signs, dtype=tensor.dtype, device=tensor.device)
    mirrored_tensor = mirrored_tensor * mirror_signs_tensor
    return mirrored_tensor

def data_augmentation(env, obs, actions, obs_type):
    if obs is None:
        obs_aug = None
    else:
        if obs_type == 'policy':
            obs_aug = torch.cat((obs, mirror_tensors(obs, POLICY_MIRROR_INDICES, POLICY_MIRROR_SIGNS)), dim=0)
        elif obs_type == 'critic':
            obs_aug = torch.cat((obs, mirror_tensors(obs, CRITIC_MIRROR_INDICES, CRITIC_MIRROR_SIGNS)), dim=0)
        else:
            raise ValueError(f"Mirror logic for observation type '{obs_type}' not implemented")
    if actions is None:
        actions_aug = None
    else:
        actions_aug = torch.cat((actions, mirror_tensors(actions, ACT_MIRROR_INDICES, ACT_MIRROR_SIGNS)), dim=0)
    return obs_aug, actions_aug

@configclass
class KuavoS42RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 50
    experiment_name = "Kuavo/s42/rough"
    empirical_normalization = True
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
    
    def _process_policy_mirror_obs(self)-> tuple[list, list]:
        idx = 0
        policy_obs_mirror_indices = []
        policy_obs_mirror_signs = []
        for k in self.policy_obs_keys:
            # 这里从类中设置的key里面获取
            n = len(singleObs[k][0])
            for i in range(self.history_len):
                start_idx = idx 
                end_idx = idx + n
                policy_obs_mirror_indices.extend([i + start_idx for i in singleObs[k][0]])
                policy_obs_mirror_signs.extend(singleObs[k][1])
                idx += n

        # policy_obs_mirror_indices = np.array(policy_obs_mirror_indices).reshape(-1)
        # policy_obs_mirror_signs = np.array(policy_obs_mirror_signs).reshape(-1)
        return policy_obs_mirror_indices, policy_obs_mirror_signs

    def _process_critic_mirror_obs(self)-> tuple[list, list]:
        idx = 0
        critic_obs_mirror_indices = []
        critic_obs_mirror_signs = []
        for k in self.critic_obs_keys:
            n = len(singleCriticObs[k][0])
            for i in range(self.history_len):
                start_idx = idx 
                end_idx = idx + n
                critic_obs_mirror_indices.extend([i + start_idx for i in singleCriticObs[k][0]])
                critic_obs_mirror_signs.extend(singleCriticObs[k][1])
                idx += n
        # critic_obs_mirror_indices = np.array(critic_obs_mirror_indices).reshape(-1)
        # critic_obs_mirror_signs = np.array(critic_obs_mirror_signs).reshape(-1)
        return critic_obs_mirror_indices, critic_obs_mirror_signs


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
        policy_obs_mirror_indices, policy_obs_mirror_signs = self._process_policy_mirror_obs()
        critic_obs_mirror_indices, critic_obs_mirror_signs = self._process_critic_mirror_obs()
        global POLICY_MIRROR_INDICES, POLICY_MIRROR_SIGNS
        global CRITIC_MIRROR_INDICES, CRITIC_MIRROR_SIGNS
        POLICY_MIRROR_INDICES = policy_obs_mirror_indices
        POLICY_MIRROR_SIGNS = policy_obs_mirror_signs
        CRITIC_MIRROR_INDICES = critic_obs_mirror_indices
        CRITIC_MIRROR_SIGNS = critic_obs_mirror_signs

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, 
            use_mirror_loss=True,
            mirror_loss_coeff=1.0, 
            data_augmentation_func=data_augmentation
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
        policy_obs_mirror_indices, policy_obs_mirror_signs = self._process_policy_mirror_obs()
        critic_obs_mirror_indices, critic_obs_mirror_signs = self._process_critic_mirror_obs()
        global POLICY_MIRROR_INDICES, POLICY_MIRROR_SIGNS
        global CRITIC_MIRROR_INDICES, CRITIC_MIRROR_SIGNS
        POLICY_MIRROR_INDICES = policy_obs_mirror_indices
        POLICY_MIRROR_SIGNS = policy_obs_mirror_signs
        CRITIC_MIRROR_INDICES = critic_obs_mirror_indices
        CRITIC_MIRROR_SIGNS = critic_obs_mirror_signs

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, 
            use_mirror_loss=True,
            mirror_loss_coeff=1.0, 
            data_augmentation_func=data_augmentation
        )
    