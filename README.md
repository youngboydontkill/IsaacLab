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
- Isaac Sim 4.2
- Isaac Lab 1.4.1
- rsl-rl 2.1.0/2.3.3

具体的安装流程参考飞书文档
### Run with Docker 
同样参考飞书文档，如果已经按照文档说明配置好了isaac-lab-base image，直接
```bash
cd docker
./run_container.sh
```
## 快速上手


### 训练配置
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