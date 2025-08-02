# Leju Kuavo Training with IsaacLab Template 

## Kuavo-S42，S46 

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


基于NVIDIA Isaacsim平台的人形机器人强化学习训练框架

## 📥 安装指南

### 环境要求
- Ubuntu 20.04/22.04 LTS
- Isaac Sim 4.5
- Isaac Lab 2.1.0
- rsl-rl 2.3.1

具体的安装流程参考飞书文档，一键安装脚本是对的
### Run with Docker 
同样参考飞书文档，如果已经按照文档说明配置好了isaac-lab-base image，直接
```bash
cd docker
./run_container.sh
```
## 快速上手


### 训练配置
**首次训练前安装exts_template和修改isaaclab加速度更新逻辑:**
1. 安装exts_template
```bash
python -m pip install -e exts/ext_template
```
2. 修改isaaclab的加速度更新逻辑
需要修改isaaclab内关于加速度更新的逻辑，在update处不更新motor的加速度，减低更新频率。但若需要观测电机加速度，不建议屏蔽此处代码。

```
#/{isaaclab_path}source/isaaclab/isaaclab/assets/articulation/articulation_data.py:98
--
    def update(self, dt: float):
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the joint acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.joint_acc
++
    def update(self, dt: float):
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the joint acceleration buffer at a higher frequency
        # since we do finite differencing.
        #self.joint_acc
```

使用VS Code调试配置进行训练：
```json
{
    "name": "Train Flat-Kuavo-S42",
    "type": "debugpy",
    "request": "launch",
    "args": [
        "--task",
        "Legged-Isaac-Velocity-Flat-Kuavo-S42-v0",
        "--num_envs",
        "4096",
        "--headless"
    ],
    "program": "${workspaceFolder}/scripts/rsl_rl/train.py",
    "console": "integratedTerminal",
}
```

或通过命令行启动：
```bash
python scripts/rsl_rl/train.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
    --num_envs 4096 \
    --headless
```

使用wandb监视训练(示例)：
```bash
python scripts/rsl_rl/train.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
    --num_envs 4096 \
    --headless \
    --logger wandb \
    --log_project_name leju-robot-rl
    --use_proxy=True
```

从checkpoint开始训练,默认会从相同task名字的最新的文件夹实验中加载模型权重，如果需要指定exp，需要设置`--load_run 2025-08-02_16-13-19`
```bash
python scripts/rsl_rl/train.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
    --num_envs 4096 \
    --headless \
    --logger wandb \
    --log_project_name leju-robot-rl
    --resume=True
    --checkpoint model_999.pt
```

### 可视化测试
播放模式配置：
```json
{
    "name": "Play Flat-Kuavo-S42",
    "type": "debugpy",
    "request": "launch",
    "args": [
        "--task",
        "Legged-Isaac-Velocity-Flat-Kuavo-S42-Play-v0",
        "--num_envs",
        "32"
    ],
    "program": "${workspaceFolder}/scripts/rsl_rl/play.py",
    "console": "integratedTerminal",
}
```

命令行启动：
```bash
python scripts/rsl_rl/play.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-Play-v0 \
    --num_envs 32
```