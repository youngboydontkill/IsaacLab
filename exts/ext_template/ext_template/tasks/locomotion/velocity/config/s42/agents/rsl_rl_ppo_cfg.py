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
policy_obs_mirror_indices = [0, 1, 2,\
                             3, 4, 5,\
                             6, 7, 8,\
                             10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31, 34, 33, \
                             36, 35, 38, 37, 40, 39, 42, 41, 44, 43, 46, 45, 48, 47, 50, 49, 52, 51, 54, 53, 56, 55, 58, 57, 60, 59, \
                             62, 61, 64, 63, 66, 65, 68, 67, 70, 69, 72, 71, 74, 73, 76, 75, 78, 77, 80, 79, 82, 81, 84, 83, 86, 85]
policy_obs_mirror_signs = [-1, 1, -1,\
                           1, -1, 1,\
                           1, -1, -1,\
                           -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, \
                           -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, \
                           -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1]
critic_obs_mirror_indices = [0, 1, 2,\
                             3, 4, 5,\
                             6, 7, 8,\
                             10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31, 34, 33, \
                             36, 35, 38, 37, 40, 39, 42, 41, 44, 43, 46, 45, 48, 47, 50, 49, 52, 51, 54, 53, 56, 55, 58, 57, 60, 59, \
                             62, 61, 64, 63, 66, 65, 68, 67, 70, 69, 72, 71, 74, 73, 76, 75, 78, 77, 80, 79, 82, 81, 84, 83, 86, 85, \
                             87, 88, 89, \
                             91, 90, 93, 92, 95, 94, 97, 96, 99, 98, 101, 100, 103, 102, 105, 104, 107, 106, 109, 108, 111, 110, 113, 112, 115, 114, \
                             117, 116, 119, 118, 121, 120, 123, 122, 125, 124, 127, 126, 129, 128, 131, 130, 133, 132, 135, 134, 137, 136, 139, 138, 141, 140, \
                             145, 146, 147, 142, 143, 144, \
                             151, 152, 153, 148, 149, 150, \
                             154, \
                             158, 159, 160, 155, 156, 157, \
                             161, 162, 163, \
                             164, \
                             165, 166, 167, \
                             168, 169, 170, \
                             172, 171, \
                             174, 173]
critic_obs_mirror_signs = [-1, 1, -1,\
                           1, -1, 1,\
                           1, -1, -1,\
                           -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, \
                           -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, \
                           -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, \
                           1, -1, 1, \
                           -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, \
                           -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, \
                           1, -1, 1, 1, -1, 1, \
                           1, -1, 1, 1, -1, 1, \
                           1, \
                           1, 1, 1, 1, 1, 1, \
                           1, -1, 1, \
                           1, \
                           1, -1, 1, \
                           -1, 1, -1, \
                           1, 1, \
                           1, 1]
act_mirror_indices = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24]
act_mirror_signs = [-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1]

# define obs&act length and check the symmetry rules, define history length
policy_single_obs_length = 87
critic_single_obs_length = 175
act_lenght = 26
history_len = 5
assert policy_single_obs_length == policy_obs_mirror_indices.__len__()
assert policy_single_obs_length == policy_obs_mirror_signs.__len__()
assert critic_single_obs_length == critic_obs_mirror_indices.__len__()
assert critic_single_obs_length == critic_obs_mirror_signs.__len__()
assert act_lenght == act_mirror_indices.__len__()
assert act_lenght == act_mirror_signs.__len__()

# expand policy obs symmetry rules w.r.t history length
singleObs = {
            "base_ang_vel":[0,3],
            "gravity":[3,6],
            "cmd":[6,9],
            "joint_pos":[9,26+9],
            "joint_vel":[26+9,26+26+9],
            "action":[26+26+9,26+26+26+9]
        }
policy_obs_mirror_indices_expanded_arranged = np.array([])    # 最终调整好顺序的下标向量
policy_obs_mirror_signs_expanded_arranged = np.array([])    # 最终调整好顺序的符号向量

policy_obs_mirror_indices_np = np.array(policy_obs_mirror_indices)
policy_obs_mirror_indices_expanded = [policy_obs_mirror_indices_np + i * policy_obs_mirror_indices_np.shape[0] for i in range(history_len)]
policy_obs_mirror_indices_expanded_np = np.array(policy_obs_mirror_indices_expanded).reshape(history_len, -1)   # 此时每行是一个
# 然后根据singleobs的chunk重新调整顺序
for k in singleObs:
    start_idx = singleObs[k][0]
    end_idx = singleObs[k][1]
    for i in range(history_len):
        obs_i = policy_obs_mirror_indices_expanded_np[i]
        # concate 
        policy_obs_mirror_indices_expanded_arranged = np.concatenate((policy_obs_mirror_indices_expanded_arranged, obs_i[start_idx:end_idx]))
        policy_obs_mirror_signs_expanded_arranged = np.concatenate((policy_obs_mirror_signs_expanded_arranged, policy_obs_mirror_signs[start_idx:end_idx]))
policy_obs_mirror_indices_expanded_arranged = np.array(policy_obs_mirror_indices_expanded_arranged).reshape(-1)
policy_obs_mirror_signs_expanded_arranged = np.array(policy_obs_mirror_signs_expanded_arranged).reshape(-1)
# print(f"policy_obs_mirror_indices_expanded_arranged:\n{policy_obs_mirror_indices_expanded_arranged}")
# print(f"policy_obs_mirror_signs_expanded_arranged:\n{policy_obs_mirror_signs_expanded_arranged}")
# print(policy_obs_mirror_signs_expanded_arranged.__len__())

# expand critic obs symmetry rules w.r.t history length
singleCri = {
            "base_ang_vel":[0,3],
            "gravity":[3,6],
            "cmd":[6,9],
            "joint_pos":[9,26+9],
            "joint_vel":[26+9,26+26+9],
            "action":[26+26+9,26+26+26+9],
            "base_lin_vel": [87, 90],
            "joint_torques": [90, 116],
            "joint_accs": [116, 142],
            "feet_lin_vel": [142, 148],
            "feet_contact_force": [148, 154],
            "base_mass_rel": [154, 155],
            "rigid_body_material": [155, 161],
            "base_com": [161, 164],
            "action_delay": [164, 165],
            "push_force": [165, 168],
            "push_torque": [168, 171],
            "feet_heights": [171, 173],
            "feet_air_times": [173, 175]
        }
critic_obs_mirror_indices_expanded_arranged = np.array([])    # 最终调整好顺序的下标向量
critic_obs_mirror_signs_expanded_arranged = np.array([])    # 最终调整好顺序的符号向量

critic_obs_mirror_indices_np = np.array(critic_obs_mirror_indices)
critic_obs_mirror_indices_expanded = [critic_obs_mirror_indices_np + i * critic_obs_mirror_indices_np.shape[0] for i in range(history_len)]
critic_obs_mirror_indices_expanded_np = np.array(critic_obs_mirror_indices_expanded).reshape(history_len, -1)   # 此时每行是一个
# 然后根据singleobs的chunk重新调整顺序
for k in singleCri:
    start_idx = singleCri[k][0]
    end_idx = singleCri[k][1]
    for i in range(history_len):
        obs_i = critic_obs_mirror_indices_expanded_np[i]
        # concate 
        critic_obs_mirror_indices_expanded_arranged = np.concatenate((critic_obs_mirror_indices_expanded_arranged, obs_i[start_idx:end_idx]))
        critic_obs_mirror_signs_expanded_arranged = np.concatenate((critic_obs_mirror_signs_expanded_arranged, critic_obs_mirror_signs[start_idx:end_idx]))
critic_obs_mirror_indices_expanded_arranged = np.array(critic_obs_mirror_indices_expanded_arranged).reshape(-1)
critic_obs_mirror_signs_expanded_arranged = np.array(critic_obs_mirror_signs_expanded_arranged).reshape(-1)
# print(f"critic_obs_mirror_indices_expanded_arranged:\n{critic_obs_mirror_indices_expanded_arranged}")
# print(f"critic_obs_mirror_signs_expanded_arranged:\n{critic_obs_mirror_signs_expanded_arranged}")
# print(critic_obs_mirror_signs_expanded_arranged.__len__())

def mirror_policy_observation(policy_obs):
    mirrored_policy_obs = policy_obs[..., policy_obs_mirror_indices_expanded_arranged]
    policy_obs_mirror_signs_tensor_expanded = torch.tensor(policy_obs_mirror_signs_expanded_arranged, 
                                                           dtype=policy_obs.dtype, 
                                                           device=policy_obs.device)
    mirrored_policy_obs = mirrored_policy_obs * policy_obs_mirror_signs_tensor_expanded
    return mirrored_policy_obs

def mirror_critic_observation(critic_obs):
    mirrored_critic_obs = critic_obs[..., critic_obs_mirror_indices_expanded_arranged]
    critic_obs_mirror_signs_tensor_expanded = torch.tensor(critic_obs_mirror_signs_expanded_arranged, 
                                                           dtype=critic_obs.dtype, 
                                                           device=critic_obs.device)
    mirrored_critic_obs = mirrored_critic_obs * critic_obs_mirror_signs_tensor_expanded
    return mirrored_critic_obs

def mirror_actions(actions):
    mirrored_actions = actions[..., act_mirror_indices]
    act_mirror_signs_tensor = torch.tensor(act_mirror_signs, dtype=actions.dtype, device=actions.device)
    mirrored_actions = mirrored_actions * act_mirror_signs_tensor
    return mirrored_actions

def data_augmentation_func(env, obs, actions, obs_type):
    if obs is None:
        obs_aug = None
    else:
        if obs_type == 'policy':
            obs_aug = torch.cat((obs, mirror_policy_observation(obs)), dim=0)
        elif obs_type == 'critic':
            obs_aug = torch.cat((obs, mirror_critic_observation(obs)), dim=0)
        else:
            raise ValueError(f"Mirror logic for observation type '{obs_type}' not implemented")
    if actions is None:
        actions_aug = None
    else:
        actions_aug = torch.cat((actions, mirror_actions(actions)), dim=0)
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
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, 
            use_mirror_loss=True,
            mirror_loss_coeff=1.0, 
            data_augmentation_func=data_augmentation_func
        ),
    )


@configclass
class KuavoS42FlatPPORunnerCfg(KuavoS42RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "Kuavo/s42/flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
