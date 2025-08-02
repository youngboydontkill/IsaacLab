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
history_len = 5

# expand policy obs symmetry rules w.r.t history length
singleObs = {
            "base_ang_vel":[[0, 1, 2],[-1,1,-1]],
            "gravity":[[0,1,2],[1,-1,1]],
            "cmd":[[0,1,2],[1, -1, -1]],
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
            "cmd":[[0,1,2],[1, -1, -1]],
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

idx = 0
policy_obs_mirror_indices = []
policy_obs_mirror_signs = []
for k in singleObs:
    n = len(singleObs[k][0])
    for i in range(history_len):
        start_idx = idx 
        end_idx = idx + n
        policy_obs_mirror_indices.extend([i+start_idx for i in singleObs[k][0]])
        policy_obs_mirror_signs.extend(singleObs[k][1])
        idx += n

policy_obs_mirror_indices = np.array(policy_obs_mirror_indices).reshape(-1)
policy_obs_mirror_signs = np.array(policy_obs_mirror_signs).reshape(-1)
# mirror critic obs
critic_obs_mirror_indices = []
critic_obs_mirror_signs = []
idx = 0
for k in singleCriticObs:
    n = len(singleCriticObs[k][0])
    for i in range(history_len):
        start_idx = idx 
        end_idx = idx + n
        critic_obs_mirror_indices.extend([i+start_idx for i in singleCriticObs[k][0]])
        critic_obs_mirror_signs.extend(singleCriticObs[k][1])
        idx += n
critic_obs_mirror_indices = np.array(critic_obs_mirror_indices).reshape(-1)
critic_obs_mirror_signs = np.array(critic_obs_mirror_signs).reshape(-1)


def mirror_policy_observation(policy_obs):
    mirrored_policy_obs = policy_obs[..., policy_obs_mirror_indices]
    policy_obs_mirror_signs_tensor_expanded = torch.tensor(policy_obs_mirror_signs, 
                                                           dtype=policy_obs.dtype, 
                                                           device=policy_obs.device)
    mirrored_policy_obs = mirrored_policy_obs * policy_obs_mirror_signs_tensor_expanded
    return mirrored_policy_obs

def mirror_critic_observation(critic_obs):
    mirrored_critic_obs = critic_obs[..., critic_obs_mirror_indices]
    critic_obs_mirror_signs_tensor_expanded = torch.tensor(critic_obs_mirror_signs, 
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
