# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL and viser web-based monitoring."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import threading
import time
from collections import deque

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL and viser monitoring.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--use_proxy", type=bool, default=False, help="Use proxy for wandb logging.")
parser.add_argument("--viser_host", type=str, default="0.0.0.0", help="Viser web server host.")
parser.add_argument("--viser_port", type=int, default=8890, help="Viser web server port.")
parser.add_argument("--viser_update_interval", type=int, default=5, help="Viser update interval in iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

if (args_cli.use_proxy):
    import os
    print("[INFO] Using proxy for wandb logging.")
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:8889"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8889"

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import statistics
import torch
import viser
import viser.transforms as vtf
from datetime import datetime
from typing import Optional

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import rsl_rl utils for store_code_state
from rsl_rl.utils import store_code_state

# Import extensions to set up environment tasks
import ext_template.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class ViserTrainingMonitor:
    """Viser-based training monitor for real-time visualization."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8890,
        max_history: int = 1000,
    ):
        """Initialize the viser monitor.

        Args:
            host: Server host address.
            port: Server port.
            max_history: Maximum number of history points to keep.
        """
        self.host = host
        self.port = port
        self.max_history = max_history

        # Data storage
        self.iterations: deque = deque(maxlen=max_history)
        self.rewards: deque = deque(maxlen=max_history)
        self.episode_lengths: deque = deque(maxlen=max_history)
        self.value_losses: deque = deque(maxlen=max_history)
        self.surrogate_losses: deque = deque(maxlen=max_history)
        self.policy_losses: deque = deque(maxlen=max_history)
        self.entropy_losses: deque = deque(maxlen=max_history)

        # State storage
        self.current_iteration = 0
        self.current_reward = 0.0
        self.current_episode_length = 0.0
        self.current_speed = [0.0, 0.0, 0.0]  # vx, vy, wz
        self.target_speed = [0.0, 0.0, 0.0]  # cmd_vx, cmd_vy, cmd_wz
        self.joint_positions: Optional[torch.Tensor] = None
        self.joint_names = []

        # Thread safety
        self.lock = threading.Lock()
        self.running = False
        self.update_event = threading.Event()

        # Start the server in a separate thread
        self._start_server()

    def _start_server(self):
        """Start the viser server in a separate thread."""

        def run_server():
            server = viser.ViserServer(host=self.host, port=self.port)
            self.server = server

            # Add UI elements
            with server.gui.add_folder("Training Status"):
                self.iteration_text = server.gui.add_text("# Iteration", initial_value="0")
                self.reward_text = server.gui.add_text("Mean Reward", initial_value="0.00")
                self.episode_len_text = server.gui.add_text("Episode Length", initial_value="0.00")
                self.speed_text = server.gui.add_text("Current Speed", initial_value="vx: 0.00, vy: 0.00, wz: 0.00")
                self.target_speed_text = server.gui.add_text("Target Speed", initial_value="vx: 0.00, vy: 0.00, wz: 0.00")

            with server.gui.add_folder("Training Curves"):
                # Reward chart
                self.reward_chart = server.gui.add_uplot(
                    data=(np.array([0.0]), np.array([0.0])),
                    series=(
                        {},  # x-axis
                        {"stroke": "rgba(0, 255, 0, 1)", "width": 2},
                    ),
                    title="Reward",
                    scales={"x": {"auto": True}, "y": {"auto": True}},
                    axes=[
                        {"label": "Iteration"},
                        {"label": "Mean Reward", "stroke": "#888"},
                    ],
                )

                # Episode length chart
                self.ep_len_chart = server.gui.add_uplot(
                    data=(np.array([0.0]), np.array([0.0])),
                    series=(
                        {},  # x-axis
                        {"stroke": "rgba(0, 0, 255, 1)", "width": 2},
                    ),
                    title="Episode Length",
                    scales={"x": {"auto": True}, "y": {"auto": True}},
                    axes=[
                        {"label": "Iteration"},
                        {"label": "Length", "stroke": "#888"},
                    ],
                )

                # Losses chart
                self.loss_chart = server.gui.add_uplot(
                    data=(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])),
                    series=(
                        {},  # x-axis
                        {"stroke": "rgba(255, 0, 0, 1)", "width": 2},
                        {"stroke": "rgba(0, 255, 255, 1)", "width": 2},
                        {"stroke": "rgba(255, 255, 0, 1)", "width": 2},
                        {"stroke": "rgba(255, 0, 255, 1)", "width": 2},
                    ),
                    title="Losses",
                    scales={"x": {"auto": True}, "y": {"auto": True}},
                    axes=[
                        {"label": "Iteration"},
                        {"label": "Loss", "stroke": "#888"},
                    ],
                )

            with server.gui.add_folder("Joint States"):
                self.joint_text = server.gui.add_text("Joint Angles", initial_value="No joint data available")

            print(f"[Viser] Server started at http://{self.host}:{self.port}")

            self.running = True
            # Update loop
            while self.running:
                self.update_event.wait(timeout=0.1)
                self.update_event.clear()
                self._update_ui()

            # Stop the server when loop exits
            server.stop()

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

    def update(
        self,
        iteration: int,
        reward: float,
        episode_length: float,
        value_loss: Optional[float] = None,
        surrogate_loss: Optional[float] = None,
        policy_loss: Optional[float] = None,
        entropy_loss: Optional[float] = None,
        speed: Optional[list] = None,
        target_speed: Optional[list] = None,
        joint_positions: Optional[torch.Tensor] = None,
        joint_names: Optional[list] = None,
    ):
        """Update the monitor with new data.

        Args:
            iteration: Current iteration number.
            reward: Mean reward.
            episode_length: Mean episode length.
            value_loss: Value function loss.
            surrogate_loss: Surrogate loss.
            policy_loss: Policy loss.
            entropy_loss: Entropy loss.
            speed: Current speed [vx, vy, wz].
            target_speed: Target speed [cmd_vx, cmd_vy, cmd_wz].
            joint_positions: Joint positions tensor.
            joint_names: Joint names list.
        """
        with self.lock:
            self.iterations.append(iteration)
            self.rewards.append(reward)
            self.episode_lengths.append(episode_length)

            if value_loss is not None:
                self.value_losses.append(value_loss)
            if surrogate_loss is not None:
                self.surrogate_losses.append(surrogate_loss)
            if policy_loss is not None:
                self.policy_losses.append(policy_loss)
            if entropy_loss is not None:
                self.entropy_losses.append(entropy_loss)

            self.current_iteration = iteration
            self.current_reward = reward
            self.current_episode_length = episode_length

            if speed is not None:
                self.current_speed = speed
            if target_speed is not None:
                self.target_speed = target_speed
            if joint_positions is not None:
                self.joint_positions = joint_positions
            if joint_names is not None:
                self.joint_names = joint_names

            self.update_event.set()

    def _update_ui(self):
        """Update the UI elements."""
        if not hasattr(self, 'server'):
            return

        with self.lock:
            # Update text elements
            self.iteration_text.value = str(self.current_iteration)
            self.reward_text.value = f"{self.current_reward:.2f}"
            self.episode_len_text.value = f"{self.current_episode_length:.2f}"
            self.speed_text.value = f"vx: {self.current_speed[0]:.2f}, vy: {self.current_speed[1]:.2f}, wz: {self.current_speed[2]:.2f}"
            self.target_speed_text.value = f"vx: {self.target_speed[0]:.2f}, vy: {self.target_speed[1]:.2f}, wz: {self.target_speed[2]:.2f}"

            # Update charts
            if len(self.iterations) > 0:
                iters = np.array(list(self.iterations))
                rewards = np.array(list(self.rewards))
                ep_lens = np.array(list(self.episode_lengths))

                self.reward_chart.data = (iters, rewards)

                self.ep_len_chart.data = (iters, ep_lens)

                if len(self.value_losses) > 0:
                    min_len = min(len(iters), len(self.value_losses))
                    self.loss_chart.data = (
                        iters[-min_len:],
                        np.array(list(self.value_losses))[-min_len:],
                        np.array(list(self.surrogate_losses))[-min_len:],
                        np.array(list(self.policy_losses))[-min_len:],
                        np.array(list(self.entropy_losses))[-min_len:],
                    )

            # Update joint states
            if self.joint_positions is not None and self.joint_names:
                joint_str = ""
                num_joints = min(len(self.joint_names), self.joint_positions.shape[-1])
                for i in range(num_joints):
                    joint_str += f"{self.joint_names[i]}: {self.joint_positions[0, i].item():.3f}\n"
                self.joint_text.value = joint_str

    def stop(self):
        """Stop the monitor."""
        self.running = False
        self.update_event.set()
        time.sleep(0.1)  # Give time for the update loop to exit


class MonitoredOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner with viser monitoring."""

    def __init__(self, *args, monitor: ViserTrainingMonitor, update_interval: int = 5, **kwargs):
        """Initialize the monitored runner.

        Args:
            monitor: Viser monitor instance.
            update_interval: Update interval in iterations.
        """
        super().__init__(*args, **kwargs)
        self.monitor = monitor
        self.update_interval = update_interval

        # Get joint names from the environment if available
        if hasattr(self.env.unwrapped, 'scene'):
            if hasattr(self.env.unwrapped.scene, 'robot'):
                if hasattr(self.env.unwrapped.scene.robot, 'data'):
                    self.joint_names = list(self.env.unwrapped.scene.robot.data.joint_names)
            else:
                self.joint_names = []
        else:
            self.joint_names = []

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Learn with viser monitoring."""
        # Initialize writer
        self._prepare_logging_writer()

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Calculate metrics
            if len(rewbuffer) > 0:
                mean_reward = statistics.mean(rewbuffer)
            else:
                mean_reward = 0.0

            if len(lenbuffer) > 0:
                mean_episode_length = statistics.mean(lenbuffer)
            else:
                mean_episode_length = 0.0

            # Extract speed information from observations if available
            speed = [0.0, 0.0, 0.0]
            target_speed = [0.0, 0.0, 0.0]
            joint_positions = None

            try:
                # Try to extract speed from observations
                obs_flat = obs
                if hasattr(obs, 'keys'):
                    obs_flat = obs['policy'] if 'policy' in obs else next(iter(obs.values()))

                obs_np = obs_flat[0].cpu().numpy() if len(obs_flat.shape) > 1 else obs_flat.cpu().numpy()

                # Try to extract base velocity and commands (common structure)
                if len(obs_np) >= 9:
                    speed = [float(obs_np[0]), float(obs_np[1]), float(obs_np[2])]  # vx, vy, wz
                    target_speed = [float(obs_np[6]), float(obs_np[7]), float(obs_np[8])]  # cmd_vx, cmd_vy, cmd_wz

                # Try to extract joint positions
                joint_start_idx = 9
                if len(obs_np) > joint_start_idx and len(self.joint_names) > 0:
                    joint_positions = torch.tensor(obs_np[joint_start_idx:joint_start_idx + len(self.joint_names)])
            except Exception as e:
                pass  # Skip if extraction fails

            # Update viser monitor
            if it % self.update_interval == 0:
                self.monitor.update(
                    iteration=it,
                    reward=mean_reward,
                    episode_length=mean_episode_length,
                    value_loss=loss_dict.get('value_loss'),
                    surrogate_loss=loss_dict.get('surrogate_loss'),
                    policy_loss=loss_dict.get('policy_loss'),
                    entropy_loss=loss_dict.get('entropy_loss'),
                    speed=speed,
                    target_speed=target_speed,
                    joint_positions=joint_positions,
                    joint_names=self.joint_names,
                )

            # Log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # Obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # If possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent and viser monitoring."""
    # Override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # Set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # Specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Initialize viser monitor
    print(f"[INFO] Starting viser monitor at http://{args_cli.viser_host}:{args_cli.viser_port}")
    monitor = ViserTrainingMonitor(
        host=args_cli.viser_host,
        port=args_cli.viser_port,
        max_history=1000,
    )

    # Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Create runner from rsl-rl
    runner = MonitoredOnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
        monitor=monitor,
        update_interval=args_cli.viser_update_interval,
    )

    # Write git state to logs
    runner.add_git_repo_to_log(__file__)
    # Save resume path before creating a new log_dir
    if agent_cfg.resume:
        # Get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # Load previously trained model
        runner.load(resume_path)

    # Dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    try:
        # Run training
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        # Stop the monitor
        monitor.stop()

    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()