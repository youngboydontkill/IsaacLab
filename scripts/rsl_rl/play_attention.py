"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np 

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from tensordict import TensorDict 
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
# for visualization 
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply,quat_apply_yaw
# Import extensions to set up environment tasks
import ext_template.tasks  # noqa: F401

import copy
from enc_actor_critic_exporter import export_enc_policy
from enc_vel_export import export_enc_vel_policy

def define_markers() -> VisualizationMarkers:
    """
    Define markers with various different shapes.
    ref: https://isaac-sim.github.io/IsaacLab/main/source/how-to/draw_markers.html
    """
    color = [mpl.colormaps['viridis'](i/9.0)[:-1] for i in range(10)]
    markeres = {}
    for i in range(10):
        markeres[f"hit_{i}"] = sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color[i])
        )
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/attention",
        markers=markeres,
    )
    return VisualizationMarkers(marker_cfg)

def visualize_attention(obs:TensorDict, attention:torch.Tensor, 
                        markers:VisualizationMarkers,ray_aligment='yaw'):
    """
    :brief : Visualize attention.
    :param obs: TensorDict, 
    :param attention: Tensor, shape (B, H, L, W)
    """
    map_scans = obs['perception']  # shape (B, H, L, W, 3)
    root_pose = obs['visualize'][:,0,...]  # shape (B, 7)
    root_pos = root_pose[:,:3]  # shape (B, 3)
    root_quat = root_pose[:,3:7]  # shape (B, 4)
    B = root_pos.shape[0]
    lastest_scan = map_scans[:,0,...]  # only visualize the lastest scan (B,L,W,3)
    L = lastest_scan.shape[1]
    W = lastest_scan.shape[2]
    # 这里换成的base frame, 所以需要最后进行一个rotation
    if (ray_aligment == 'yaw'):
        last_scan_flatten = lastest_scan.reshape(-1, 3)
        height_scan = quat_apply_yaw(root_quat.repeat(1, L*W), last_scan_flatten)
        lastest_scan = height_scan.view(B,L,W,3)
    elif (ray_aligment == 'base'):
        last_scan_flatten = lastest_scan.view(-1,3)  # shape (B*L*W, 3)
        height_scan = quat_apply(root_quat.repeat(1, L*W), last_scan_flatten)
        lastest_scan = height_scan.view(B,L,W,3)
    lastest_scan_world = -lastest_scan + root_pos.unsqueeze(1).unsqueeze(1)  # shape (B, L, W, 3)
    lastest_attention = attention[:,0,...]  # only visualize the lastest attention
    # print(torch.sum(lastest_attention,dim=(1,2)))
    # 首先求取batch内最大的attention，然后归一化
    for env_idx in range(lastest_attention.size(0)):
        lastest_attention[env_idx, :] = lastest_attention[env_idx, :] / (torch.max(lastest_attention[env_idx, :]) + 1e-8)
    # lastest_attention = lastest_attention.view(-1)
    lastest_attention = lastest_attention.reshape(-1)
    # print(torch.max(lastest_attention))
    attention_indices = torch.zeros_like(lastest_attention,dtype=torch.int)
    for i in range(10):
        color_mask = (lastest_attention > 0.1*i) 
        color_mask = torch.bitwise_and(color_mask, lastest_attention < 0.1*(i+1))
        attention_indices[color_mask] = i
    markers.visualize(translations=lastest_scan_world.view(-1,3),marker_indices=attention_indices)

def main():
    """Play with RSL-RL agent."""
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
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    combined_obs_keys = agent_cfg.command_obs_keys + agent_cfg.policy_obs_keys
    export_enc_policy(
        ppo_runner.alg.policy,obs=combined_obs_keys, path=export_model_dir, filename="enc_policy_s45.onnx"
    )
    export_enc_vel_policy(
        ppo_runner.alg.policy,obs=combined_obs_keys, path=export_model_dir, filename="enc_vel_policy_s45.onnx"
    )
    # create markers :
    visualizer = define_markers()
    # reset environment
    obs = env.get_observations()
    timestep = 0
    simulation_steps = 100000
    sim_step = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions,attention = policy(obs)  # only for attention!
            # visualize attention
            visualize_attention(obs,attention,visualizer)
            # env stepping
            obs, _, _, _ = env.step(actions)
            sim_step += 1
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if (sim_step == simulation_steps):
            print("Simulation steps exceeded")
            break

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
