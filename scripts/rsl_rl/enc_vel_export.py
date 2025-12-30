"""
Brief : Exporter for observation with shape = [B,H,D,...]
"""

import argparse
import numpy as np 
import os
import torch
import copy
from tensordict import TensorDict 

OBSTERMLAB2GYM = {
            "base_ang_vel":[0,1,2],
            "gravity":[0,1,2],
            "cmd":[0,1,2,3],
            "cmd_vel":[0,1,2],
            "hugwbc_cmd":[0],
            "joint_pos":[0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 24, 3, 7, 11, 15, 19, 23, 25],
            "joint_vel":[0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 24, 3, 7, 11, 15, 19, 23, 25],
            "action":[0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 24, 3, 7, 11, 15, 19, 23, 25],
            "base_lin_vel": [0, 1, 2],
            "height_scan": [i for i in range(187)],
            "joint_torques": [0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 24, 3, 7, 11, 15, 19, 23, 25],
            "joint_accs": [0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 24, 3, 7, 11, 15, 19, 23, 25],
            "feet_lin_vel": [i for i in range(6)],
            "feet_contact_force": [i for i in range(6)],
            "base_mass_rel": [0],
            "rigid_body_material": [i for i in range(6)],
            "base_com": [[0,1,2],],
            "action_delay": [0,],
            "push_force": [0,1,2],
            "push_torque": [0, 1, 2],
            "feet_heights": [0,1],
            "feet_air_times": [0,1],
        }
OBSTERMGYM2LAB = {
            "base_ang_vel":[0,1,2],
            "gravity":[0,1,2],
            "cmd":[0,1,2,3],
            "cmd_vel":[0,1,2],
            "hugwbc_cmd":[0],
            "joint_pos":[0, 6, 12, 19, 1, 7, 13, 20, 2, 8, 14, 21, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25],
            "joint_vel":[0, 6, 12, 19, 1, 7, 13, 20, 2, 8, 14, 21, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25],
            "action":[0, 6, 12, 19, 1, 7, 13, 20, 2, 8, 14, 21, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25],
            "base_lin_vel": [0, 1, 2],
            "height_scan": [i for i in range(187)],
            "joint_torques": [0, 6, 12, 19, 1, 7, 13, 20, 2, 8, 14, 21, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25],
            "joint_accs": [0, 6, 12, 19, 1, 7, 13, 20, 2, 8, 14, 21, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25],
            "feet_lin_vel": [i for i in range(6)],
            "feet_contact_force": [i for i in range(6)],
            "base_mass_rel": [0],
            "rigid_body_material": [i for i in range(6)],
            "base_com": [[0,1,2],],
            "action_delay": [0,],
            "push_force": [0,1,2],
            "push_torque": [0, 1, 2],
            "feet_heights": [0,1],
            "feet_air_times": [0,1],
        }

def generate_lab_obs_indices(obs_keys:dict,history_len:int)->np.array:
    """
    将gym的风格的观测转换为lab的索引 obs[indices]->[obs_1[0:h],obs_2[0:h],...]
    """
    policy_obs_idx = 0 
    single_gym2lab_dict = {}
    single_gym2lab = np.array([])
    # 先构建(history, N)的矩阵，然后将其改为列存储
    for k in obs_keys:
        term_gym2lab = np.array(OBSTERMGYM2LAB[k])
        n = term_gym2lab.shape[0]
        single_gym2lab_dict[k] = [policy_obs_idx, policy_obs_idx+n]
        single_gym2lab = np.concatenate((single_gym2lab, term_gym2lab+policy_obs_idx), axis=-1)
        policy_obs_idx += n
    gym2lab_mat = [single_gym2lab + i * single_gym2lab.shape[0] for i in range(history_len)]
    gym2lab_mat = np.array(gym2lab_mat).reshape(history_len,-1)  # 此时每行是一个
    gym2lab = np.array([])

    for k in obs_keys:
        start_idx = single_gym2lab_dict[k][0]
        end_idx = single_gym2lab_dict[k][1]
        for i in range(history_len):
            obs_i = gym2lab_mat[i]
            gym2lab = np.concatenate((gym2lab, obs_i[start_idx:end_idx]))
    gym2lab = np.array(gym2lab).reshape(-1)
    return gym2lab.astype(np.int32)

def generate_gym_obs_indices(obs_keys:dict,history_len:int)->np.array:
    """
    生成gym风格的观测索引 obs[indices]-> [obs_1,obs_2,...,obs_n]
    """
    single_lab2gym = np.array([])
    single_add_indices = np.array([])
    # 先构建(history, N)的矩阵，然后将其改为列存储
    skip_indices = 0
    for k in obs_keys:
        term_lab2gym = np.array(OBSTERMLAB2GYM[k])
        n = term_lab2gym.shape[0]  # 该观测项的维度
        single_lab2gym = np.concatenate((single_lab2gym, term_lab2gym+skip_indices), axis=-1)
        single_add_indices = np.concatenate((single_add_indices, np.array([n]*n)), axis=-1)
        skip_indices += n*history_len
    lab2gym = [single_lab2gym + single_add_indices*i for i in range(history_len)]
    lab2gym = np.array(lab2gym).reshape(-1)
    return lab2gym.astype(np.int32)

def export_enc_vel_policy(
    actor_critic: object, path: str, obs:dict, filename="policy.onnx", verbose=False
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = EncVelActorCriticExporter(actor_critic, obs, verbose)
    policy_exporter.export(path, filename)


def export_velocity_estimator(
    actor_critic: object,
    path: str,
    obs: dict,
    filename: str = "velocity_estimator.onnx",
    verbose: bool = False,
):
    """Export velocity estimator as a standalone ONNX.

    Notes:
        - The exported model only depends on the low-dim proprioceptive history.
        - Input is a flattened tensor that contains ONLY the history prop (5*91 by default).
          This mirrors the layout used inside :class:`EncVelActorCriticExporter` but keeps the
          interface minimal for deployment.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    vel_exporter = VelocityEstimatorExporter(actor_critic, obs, verbose)
    vel_exporter.export(path, filename)


class EncVelActorCriticExporter(torch.nn.Module):
    """Exporter of actor-critic with velocity estimator into ONNX file.
    
    输入 (展平为单一输入):
        - obs: [B, prop_current + map_scan_flatten + prop_history_flatten]
               = [B, 91 + L*W*C + 5*91]
               = [B, 当前本体感觉(91) | map_scan(L*W*C) | 历史本体感觉(5*91)]
    
    输出:
        - actions: [B, num_actions] 动作输出
        - vel_est: [B, 3] 速度估计输出
    
    处理流程:
        1. 从展平的obs中解析出prop_current, map_scan, prop_history
        2. 从5帧prop_history中去掉速度维度(4:7)，得到[B, 5, 88]
        3. 将去掉速度的prop输入velocity_estimator得到vel_est [B, 3]
        4. 用vel_est替换当前帧prop中的速度(4:7)
        5. 将替换后的prop和map_scan输入encoder和actor得到action
        6. 返回action和vel_est
    """
    # 因为在对称性增强时将command分离了出来：policy_obs_key中少了4维度。
    def __init__(self, actor_critic, obs_keys:dict, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor_critic = copy.deepcopy(actor_critic)
        self.obs_groups = actor_critic.obs_groups
        self.encoder_history = 1  # encoder只需要1帧
        self.vel_estimator_history = 5  # 速度估计器需要5帧
        self.policy_obs_keys = obs_keys
        
        # 本体感觉维度配置
        self.prop_dim_with_vel = 91  # 包含速度的prop维度
        self.prop_dim_without_vel = 84  # 不包含速度的prop维度（用于vel_estimator输入） 去除command
        self.vel_dim = 3  # 速度维度
        self.vel_start_idx = 4  # 速度在prop中的起始索引
        self.vel_end_idx = 7  # 速度在prop中的结束索引
        
        # map_scan维度
        self.map_scan_shape = actor_critic.high_dim_obs_shape[2:]  # [L, W, C]
        self.map_scan_flatten_dim = int(np.prod(self.map_scan_shape))
        
        # 计算各部分在展平输入中的位置
        # obs布局: [prop_current(91) | map_scan(L*W*C) | prop_history(5*91)]
        self.prop_current_start = 0
        self.prop_current_end = self.prop_dim_with_vel  # 91
        self.map_scan_start = self.prop_current_end  # 91
        self.map_scan_end = self.map_scan_start + self.map_scan_flatten_dim
        self.prop_history_start = self.map_scan_end
        self.prop_history_end = self.prop_history_start + self.vel_estimator_history * self.prop_dim_with_vel  # 5*91
        
        # 总输入维度
        self.total_obs_dim = self.prop_history_end
        
        # 根据key构造gym2lab (针对包含速度的91维)
        self.gym2lab = generate_lab_obs_indices(self.policy_obs_keys, 1)
        self.lab2gym = [0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 24, 3, 7, 11, 15, 19, 23, 25]
        
        # 获取num_actor_obs
        self.num_actor_obs = actor_critic.num_actor_obs  # 91

    def _remove_velocity_from_prop(self, prop: torch.Tensor) -> torch.Tensor:
        """
        从prop中移除速度维度 [B, H, 91] -> [B, H, 84]
        速度位于索引 4:7
        """
        # prop_without_vel = torch.cat([
        #     prop[:, :, :self.vel_start_idx],  # [B, H, 4] 前4维
        #     prop[:, :, self.vel_end_idx:]     # [B, H, 84] 后84维
        # ], dim=-1)  # [B, H, 88]
        # return prop_without_vel
        prop_without_vel = prop[:,:,self.vel_end_idx:]
        return prop_without_vel

    def _replace_velocity_in_prop(self, prop: torch.Tensor, vel_est: torch.Tensor) -> torch.Tensor:
        """
        用估计的速度替换prop中的速度 
        prop: [B, 1, 91], vel_est: [B, 3] -> [B, 1, 91]
        """
        prop_with_new_vel = torch.cat([
            prop[:, :, :self.vel_start_idx],  # [B, 1, 4] 前4维
            vel_est.unsqueeze(1),              # [B, 1, 3] 估计的速度
            prop[:, :, self.vel_end_idx:]     # [B, 1, 84] 后84维
        ], dim=-1)  # [B, 1, 91]
        return prop_with_new_vel

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: [B, total_obs_dim] 展平的观测
                 布局: [prop_current(91) | map_scan(L*W*C) | prop_history(5*91)]
        
        Returns:
            action: [B, num_actions] 动作输出
            vel_est: [B, 3] 速度估计输出
        """
        B = obs.shape[0]
        
        # Step 1: 从展平的obs中解析各部分
        prop_current_flat = obs[:, self.prop_current_start:self.prop_current_end]  # [B, 91]
        map_scan_flat = obs[:, self.map_scan_start:self.map_scan_end]  # [B, L*W*C]
        prop_history_flat = obs[:, self.prop_history_start:self.prop_history_end]  # [B, 5*91]
        
        # Step 2: 重塑为所需形状
        prop_current = prop_current_flat.unsqueeze(1)  # [B, 1, 91]

        L, W, C = self.map_scan_shape  # 目标形状
        # 输入的  [B,1,W,L,3]
        map_scan = map_scan_flat.view(B, 1, W, L, C).transpose(2, 3)  # [B, 1, L, W, C]
        # map_scan = map_scan_flat.view(B, 1, *self.map_scan_shape)  # [B, 1, L, W, C]
        prop_history = prop_history_flat.view(B, self.vel_estimator_history, self.prop_dim_with_vel)  # [B, 5, 91]

        # Step 3: 应用gym2lab索引转换
        lab_prop_current = prop_current[:, :, self.gym2lab]  # [B, 1, 91]
        lab_prop_history = prop_history[:, :, self.gym2lab]  # [B, 5, 91]
        
        # Step 4: 从5帧prop中移除速度维度，用于速度估计
        prop_history_without_vel = self._remove_velocity_from_prop(lab_prop_history)  # [B, 5, 88]
        
        # Step 5: 使用去掉速度的5帧prop进行速度估计
        vel_est = self.actor_critic.velocity_estimator(prop_history_without_vel)  # [B, 3]
        
        # Step 6: 用估计的速度替换当前帧prop中的速度
        prop_with_est_vel = self._replace_velocity_in_prop(lab_prop_current, vel_est)  # [B, 1, 91]
        
        
        # Step 7: 归一化
        low_dim_obs = self.actor_critic.actor_obs_normalizer(prop_with_est_vel)  # [B, 1, 91]
        
        # Step 8: 计算embedding
        embedding, attention = self.actor_critic.encoder(
            map_scan, low_dim_obs, embedding_only=False
        )
        embedding_vec = embedding.view(B, -1)  # [B, 1*(embedding_dim + d_obs)]
        
        # Step 9: 计算action
        action = self.actor_critic.actor(embedding_vec)
        
        # 返回action和速度估计
        return action[:, self.lab2gym], vel_est
    
    def export(self, path, filename):
        self.to("cpu")
        
        # 创建示例输入
        # obs: [B, prop_current + map_scan_flatten + prop_history_flatten]
        obs = torch.randn(1, self.total_obs_dim)
        
        torch.onnx.export(
            self,
            (obs,),
            os.path.join(path, filename),
            export_params=True,
            opset_version=14,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions", "vel_est"],
            dynamic_axes={},
        )
        
        print(f"[INFO] Exported policy to {os.path.join(path, filename)}")
        print(f"  Input:")
        print(f"    - obs shape: [B, {self.total_obs_dim}]")
        print(f"    - Layout: [prop_current({self.prop_dim_with_vel}) | map_scan({self.map_scan_flatten_dim}) | prop_history({self.vel_estimator_history}*{self.prop_dim_with_vel}={self.vel_estimator_history * self.prop_dim_with_vel})]")
        print(f"    - Indices: prop_current[{self.prop_current_start}:{self.prop_current_end}], map_scan[{self.map_scan_start}:{self.map_scan_end}], prop_history[{self.prop_history_start}:{self.prop_history_end}]")
        print(f"  Outputs:")
        print(f"    - actions shape: [B, num_actions]")
        print(f"    - vel_est shape: [B, {self.vel_dim}]")
        print(f"  Internal:")
        print(f"    - Velocity estimator input: [B, {self.vel_estimator_history}, {self.prop_dim_without_vel}] (velocity removed)")
        print(f"    - Velocity position in prop: [{self.vel_start_idx}:{self.vel_end_idx}]")


class VelocityEstimatorExporter(torch.nn.Module):
    """Standalone exporter for :attr:`actor_critic.velocity_estimator`.

    Input (flattened):
        - prop_history_flat: [B, vel_estimator_history * prop_dim_with_vel]
          Default: [B, 5*91]

    Output:
        - vel_est: [B, 3]

    Internal:
        - Applies the same gym->lab index mapping as training/export.
        - Removes velocity dims (4:7) to match estimator input (d_obs=84 in your current policy).
    """

    def __init__(self, actor_critic, obs_keys: dict, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.actor_critic = copy.deepcopy(actor_critic)

        # Keep these in sync with EncVelActorCriticExporter
        self.vel_estimator_history = 5
        self.prop_dim_with_vel = 91
        self.vel_start_idx = 4
        self.vel_end_idx = 7

        # estimator input dim after removing velocity dims (and, in your current setup, removing command)
        # Here we mimic EncVelActorCriticExporter._remove_velocity_from_prop implementation.
        self.prop_dim_without_vel = 84

        # Indices mapping: gym -> lab for a single frame (91-dim)
        self.policy_obs_keys = obs_keys
        self.gym2lab = generate_lab_obs_indices(self.policy_obs_keys, 1)

        self.total_input_dim = self.vel_estimator_history * self.prop_dim_with_vel

    def _remove_velocity_from_prop(self, prop: torch.Tensor) -> torch.Tensor:
        # Keep consistent with EncVelActorCriticExporter
        return prop[:, :, self.vel_end_idx:]

    def forward(self, prop_history_flat: torch.Tensor):
        """Args:
        prop_history_flat: [B, 5*91] flattened history in *gym* term ordering.
        """
        B = prop_history_flat.shape[0]
        prop_history = prop_history_flat.view(B, self.vel_estimator_history, self.prop_dim_with_vel)
        # convert gym ordering to lab ordering for each frame
        lab_prop_history = prop_history[:, :, self.gym2lab]
        prop_history_without_vel = self._remove_velocity_from_prop(lab_prop_history)
        vel_est = self.actor_critic.velocity_estimator(prop_history_without_vel)
        return vel_est

    def export(self, path: str, filename: str):
        self.to("cpu")
        example_inp = torch.randn(1, self.total_input_dim)
        out_path = os.path.join(path, filename)
        torch.onnx.export(
            self,
            (example_inp,),
            out_path,
            export_params=True,
            opset_version=14,
            verbose=self.verbose,
            input_names=["prop_history"],
            output_names=["vel_est"],
            dynamic_axes={"prop_history": {0: "B"}, "vel_est": {0: "B"}},
        )
        print(f"[INFO] Exported velocity_estimator to {out_path}")
        print(f"  Input: prop_history shape [B, {self.total_input_dim}] (layout: 5*91 flattened)")
        print(f"  Output: vel_est shape [B, 3]")


# 保留原来的 EncActorCriticExporter 类以备用
class EncActorCriticExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file (without velocity estimator)."""
    # ...existing code...
    def __init__(self, actor_critic, obs_keys:dict, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor_critic = copy.deepcopy(actor_critic)
        self.obs_groups = actor_critic.obs_groups
        self.encoder_history = 1
        self.vel_estimator_history = 5
        self.policy_obs_keys = obs_keys
        self.prop_dim_without_vel = 88
        self.vel_dim = 3
        self.vel_insert_idx = 4
        self.gym2lab = generate_lab_obs_indices(self.policy_obs_keys, 1)
        self.lab2gym = [0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 24, 3, 7, 11, 15, 19, 23, 25]
        self.num_actor_obs = actor_critic.num_actor_obs

    def forward(self, prop_history: torch.Tensor, obs_current: torch.Tensor):
        # ...existing code...
        pass
    
    def export(self, path, filename):
        # ...existing code...
        pass


if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    # local imports
    import cli_args  # isort: skip

    # add argparse arguments
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    # from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    import gymnasium as gym
    from isaaclab.utils.dict import print_dict
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlVecEnvWrapper,
    )
    from rsl_rl.runners import OnPolicyRunner
    # Import extensions to set up environment tasks
    import ext_template.tasks  # noqa: F401
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_enc_vel_policy(
        ppo_runner.alg.policy, obs=agent_cfg.policy_obs_keys, path=export_model_dir, filename="enc_vel_policy.onnx"
    )

    # Export velocity estimator only (standalone)
    export_velocity_estimator(
        ppo_runner.alg.policy, obs=agent_cfg.policy_obs_keys, path=export_model_dir, filename="velocity_estimator.onnx"
    )