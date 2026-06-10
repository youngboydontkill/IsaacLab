import argparse
import numpy as np 
import os
import torch
import copy

S54_URDF2USD = [0, 6, 12, 1, 7, 13, 20, 2, 8, 14, 21, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26]
S54_USD2URDF = [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 23, 25, 6, 10, 14, 18, 22, 24, 26]

OBSTERMLAB2GYM = {
            "base_ang_vel":[0,1,2],
            "gravity":[0,1,2],
            "cmd":[0,1,2,3],
            "cmd_vel":[0,1,2],
            "hugwbc_cmd":[0],
            # 0 - 26
            "joint_pos": [i for i in range(27)],
            "joint_vel": [i for i in range(27)],
            "action": S54_USD2URDF,

            "base_lin_vel": [0, 1, 2],
            "height_scan": [i for i in range(187)],
            # 0 -> 26
            "joint_torques": [i for i in range(27)],
            "joint_accs": [i for i in range(27)],
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
            "joint_pos": [i for i in range(27)],
            "joint_vel": [i for i in range(27)],
            "action": S54_URDF2USD,
            "base_lin_vel": [0, 1, 2],
            # 高程图，1.6m x 1.0m，分辨率0.1m，共17x11个点，排列顺序为xy，所以关于y对称就是每隔17个点为一列，把这11列倒序排列即可。符号不变。
            "height_scan": [i for i in range(187)],
            "joint_torques":  [i for i in range(27)],
            "joint_accs": [i for i in range(27)],
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

def export_policy_as_onnx_s42(
    actor_critic: object, path: str, obs:dict, normalizer: object | None = None,
    filename="policy.onnx", verbose=False, history_len:int=5
):
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        obs : actor dict keys from cfg
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
        history_len: History length for observation stacking. Defaults to 5.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, obs, history_len, normalizer, verbose)
    policy_exporter.export(path, filename)

def export_policy_as_onnx_s54(
    actor_critic: object, path: str, obs:dict, normalizer: object | None = None,
    filename="policy.onnx", verbose=False, history_len:int=5
):
    """Export S54 policy into ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        obs: Policy observation keys from agent config.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
        history_len: History length for observation stacking. Defaults to 5.

    Note:
        S54 uses identity mapping for action (IsaacLab and MuJoCo joint order are consistent).
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, obs, history_len, normalizer, verbose)
    policy_exporter.export(path, filename)
    
class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, actor_critic, obs:dict, history_len:int=5, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        self.policy_obs_keys = obs
        self.history_len = history_len
        # 根据key构造gym2lab: 将 gym style 转换为 lab style
        self.gym2lab = generate_lab_obs_indices(self.policy_obs_keys, self.history_len)
        # S54: identity 映射 (IsaacLab 和 MuJoCo 关节顺序一致)
        self.lab2gym = OBSTERMLAB2GYM["action"]

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x[:, self.gym2lab])[:, self.lab2gym], h, c

    def forward(self, x):
        return self.actor(self.normalizer(x[:, self.gym2lab]))[:, self.lab2gym]

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )

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
    # convert to single-agent instance if required by the RL algorithm
    # if isinstance(env.unwrapped, DirectMARLEnv):
    #     env = multi_agent_to_single_agent(env)

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
    export_policy_as_onnx_s42(
        ppo_runner.alg.policy,obs=agent_cfg.policy_obs_keys,normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy_s45.onnx"
    )