"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import matplotlib.pyplot as plt 
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

# Import extensions to set up environment tasks
import ext_template.tasks  # noqa: F401

import copy
from exporter import export_policy_as_onnx_s42,generate_gym_obs_indices,OBSTERMLAB2GYM


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
    # export_policy_as_jit(
    #     ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )
    export_policy_as_onnx_s42(
        ppo_runner.alg.policy,obs=agent_cfg.policy_obs_keys,normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy_s45.onnx"
    )
    # 生成lab转gym的policy观测索引
    history_len = agent_cfg.history_len
    policy_lab2gym = generate_gym_obs_indices(agent_cfg.policy_obs_keys,agent_cfg.history_len)
    action_lab2gym = np.array(OBSTERMLAB2GYM["action"])

    # asset = env.scene["robot"]
    # print(asset.data.joint_names)
    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    record_obs = []
    record_action = []
    simulation_steps = 1000
    sim_step = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            obs_np = obs.cpu().detach().numpy()
            action_np = actions.cpu().detach().numpy()
            obs_i = obs_np[:,policy_lab2gym]
            act_i = action_np[action_lab2gym]
            record_obs.append(obs_i)
            record_action.append(act_i)
            # print(record_obs[-1][0,:15])
            # if (len(record_obs) > 1):
                # print(record_obs[-1][0,0:6] - record_obs[-2][0,3:9])
                # print(record_obs[-1][0,3:9] - record_obs[-2][0,:6])
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
    # plot record data 
    joint_names = ['leg_l1_joint', 'leg_r1_joint', 'zarm_l1_joint', 'zarm_r1_joint',
                    'leg_l2_joint', 'leg_r2_joint', 'zarm_l2_joint', 'zarm_r2_joint',
                      'leg_l3_joint', 'leg_r3_joint', 'zarm_l3_joint', 'zarm_r3_joint',
                        'leg_l4_joint', 'leg_r4_joint', 'zarm_l4_joint', 'zarm_r4_joint', 
                        'leg_l5_joint', 'leg_r5_joint', 'zarm_l5_joint', 'zarm_r5_joint',
                          'leg_l6_joint', 'leg_r6_joint', 'zarm_l6_joint', 'zarm_r6_joint', 
                          'zarm_l7_joint', 'zarm_r7_joint']
    print("Plotting record data")
    robot_id = 0
    record_obs_np = np.array(record_obs)
    record_action_np = np.array(record_action)
    n_single_obs = int(record_obs_np.shape[-1] / history_len)
    record_robot_obs = record_obs_np[:, robot_id, -n_single_obs:]  # lastest obs 
    record_robot_action = record_action_np[:, robot_id, :]
    vx = record_robot_obs[:, 0]
    vy = record_robot_obs[:, 1]
    wz = record_robot_obs[:, 2]
    cmd_vx = record_robot_obs[:,6]
    cmd_vy = record_robot_obs[:,7]
    cmd_wz = record_robot_obs[:,8]
    fig = plt.figure()
    ax = fig.add_subplot(221)
    # ax.plot(vx, label="vx")
    # ax.plot(vy, label="vy")
    ax.plot(wz, label="wz")
    ax.plot(cmd_vx, label="cmd_vx")
    ax.plot(cmd_vy, label="cmd_vy")
    ax.plot(cmd_wz, label="cmd_wz")
    plt.legend()
    ax = fig.add_subplot(222)
    plt.plot(record_robot_obs[:, 9+3], label="left knee")
    plt.plot(record_robot_action[:, 3]*0.25, label="left knee req")
    # plt.plot(record_robot_obs[:, 9+6+3], label="right knee")
    # plt.plot(record_robot_action[:, 6+3]*0.25, label="right knee req")
    plt.legend()
    ax = fig.add_subplot(223)
    plt.plot(record_robot_obs[:, 9+0], label="left hip roll")
    plt.plot(record_robot_action[:, 0]*0.25, label="left hip roll req")
    # plt.plot(record_robot_obs[:, 9+6+0], label="right hip roll")
    # plt.plot(record_robot_action[:, 6+0]*0.25, label="right hip roll req")
    plt.legend()
    ax = fig.add_subplot(224)
    # plt.plot(record_robot_obs[:, 9+3], label="left knee")
    # plt.plot(record_robot_obs[:, 9+2], label="left hip")
    # plt.plot(record_robot_obs[:, 9+4], label="left ank")
    # plt.plot(record_robot_obs[:, 9+3]+record_robot_obs[:, 9+2]+record_robot_obs[:, 9+4], label="left total")
    plt.plot(record_robot_obs[:, 3], label="g_x")
    plt.plot(record_robot_obs[:, 4], label="g_y")
    plt.plot(record_robot_obs[:, 5], label="g_z")
    # plt.plot(record_robot_obs[:, 9+1], label="left hip yaw")
    # plt.plot(record_robot_action[:, 1]*0.25, label="left hip yaw req")
    # plt.plot(record_robot_obs[:, 9+6+1], label="right hip yaw")
    # plt.plot(record_robot_action[:, 6+1]*0.25, label="right hip yaw req")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
