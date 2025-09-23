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
使用vscode调试建议安装`Data Wrangler`插件用于可视化CPU的Tensor

命令行启动：
```bash
python scripts/rsl_rl/play.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-Play-v0 \
    --num_envs 32
```
启动HugWBC的play:
```bash
python scripts/rsl_rl/play.py \
    --task Legged-Isaac-Velocity-Flat-HugWBC-Kuavo-S42-Play-v0 \
    --num_envs 32
```
启动Attention Based 的play
```bash
python scripts/rsl_rl/play_attention.py \
    --task Legged-Isaac-Attention-Rough-Kuavo-S42-Play-v0 \
    --num_envs 32
```
### 迁移到RSL RL 3.1
由于图像或scan height等高维度观测的输入，一般需要在policy中加入一个encoder,这就使得actor和critic会共用一个embedding的输入，原始的rsl-rl的`VecEnv.step`定义返回的`[torch.Tensor,torch.Tensor,torch.Tensor,dict]`将actor的obs固定在了一个Tensor之中,不好拆分进行分别处理. 因此在rsl-rl 3.0.1版本之后将返回值修改为`[TensorDict,torch.Tensor,torch.Tensor,dict]`,这方便我们在ActorCritic同级的类中进行拆分处理. 要从历史版本的训练代码迁移到新版本,需要注意以下几点
1. `ObservationCfg`
   现在的`ObservationCfg`中支持定义一堆`ObsGroup`,其都会在TensorDict中返回,例如:
   ```python
    @configclass
        class ObservationsCfg:
        """Observation specifications for the MDP."""

        @configclass
        class PerceptionCfg(ObsGroup):
            flatten_history_dim = False

        @configclass
        class PolicyCfg(ObsGroup):
            flatten_history_dim = False

        @configclass
        class PrivilegedCfg(ObsGroup):
            flatten_history_dim = False
        # observation groups
        policy: PolicyCfg = PolicyCfg()
        privileged: PrivilegedCfg = PrivilegedCfg()
        perception: PerceptionCfg = PerceptionCfg()
   ```
   此时环境返回的`TensorDict`中会包含`policy`,`privileged`,`perception`三个key,分别对应`PolicyCfg`,`PrivilegedCfg`,`PerceptionCfg`中定义的obs. 需要特别注意的,原有的`CriticCfg(PolicyCfg)`以及`critic = CriticCfg()`需要修改为上述的写法,特别是需要注意这里不需要进行继承.
   这里的`flatten_history_dim`表示是否将历史维度展平,如果为True,则返回的尺寸为[B*H,D],否则为[B,H,D,...],这里建议选择False,具体原因后面说.
2. `RslRlOnPolicyRunnerCfg` \
    上面的`ObservationCfg`定义了环境返回的`TensorDict`中包含的key,而`RslRlOnPolicyRunnerCfg`则定义了如何使用这些key, 例如:
    ```python
    @configclass
    class KuavoS42RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
        # empirical_normalization = True
        obs_groups = {
            "policy": ["policy"],
            "critic": ["policy", "privileged"],
        }
        policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            actor_obs_normalization=True 
            critic_obs_normalization=True
        )
    
    ```
    这里新增了`obs_groups`的属性,其中key `policy`代表actor需要使用的obs,`critic`代表critic需要使用的obs,`critic`对应的输入在进行计算时会在最后一个维度进行concat. 由于isaaclab在`flatten_history_dim=True`时会神奇的将历史观测进行列优先存储,此时concat得到顺序会很怪,因此建议`flatten_history_dim=False`.\
    另外, 在新版rsl rl中移除了`empirical_normalization`属性,如果需要使用经验归一化,可以在`policy`中设置`actor_obs_normalization`和`critic_obs_normalization`属性,具体参考`RslRlPpoActorCriticCfg`的文档.
3. `data_augmentation`.
   由于`obs`的类型变了,因此对称性增强的函数原型也发生了变化,这里将对称性增强的相关代码都集成到了一个python文件中,方便使用. 只需要根据前面的cfg设置好key就可以了, 只需要记得`flatten_history_dim=True`使用`SymmetryAug2D`,`flatten_history_dim=False`使用`SymmetryAug`.即可.