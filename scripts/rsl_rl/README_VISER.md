# Viser训练监控系统 - 使用指南

## 概述

本系统提供了基于viser的实时训练可视化监控功能，可以在训练过程中实时查看训练指标、损失曲线和机器人状态。

## 已创建的文件

1. **`scripts/rsl_rl/train_with_viser.py`** - 带viser监控的训练脚本
2. **`scripts/rsl_rl/test_viser.py`** - viser功能测试脚本
3. **`scripts/rsl_rl/VISER_TRAINING.md`** - 详细使用文档

## 快速开始

### 1. 安装依赖

viser已经安装成功。如果需要重新安装：

```bash
pip install viser
```

### 2. 测试viser功能

```bash
cd IsaacLab/scripts/rsl_rl
python test_viser.py --port 8891
```

然后在浏览器中访问 `http://localhost:8891` 查看测试界面。

### 3. 运行带viser监控的训练

```bash
cd IsaacLab/scripts/rsl_rl
python train_with_viser.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
    --num_envs 4096 \
    --viser_port 8890
```

然后在浏览器中访问 `http://localhost:8890` 查看训练监控界面。

## 功能特性

### 实时监控界面

训练监控界面包含以下几个部分：

#### 1. Training Status（训练状态）
- **Iteration**: 当前训练迭代次数
- **Mean Reward**: 平均奖励
- **Episode Length**: 平均回合长度
- **Current Speed**: 机器人当前速度 (vx, vy, wz)
- **Target Speed**: 目标速度 (cmd_vx, cmd_vy, cmd_wz)

#### 2. Training Curves（训练曲线）
- **Reward**: 奖励曲线，显示平均奖励随迭代的变化
- **Episode Length**: 回合长度曲线
- **Losses**: 损失曲线，包含：
  - Value Loss（价值函数损失）
  - Surrogate Loss（替代损失）
  - Policy Loss（策略损失）
  - Entropy Loss（熵损失）

#### 3. Joint States（关节状态）
- 实时显示机器人各关节的角度值

## 命令行参数

### 基础参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务名称 | - |
| `--num_envs` | 环境数量 | - |
| `--max_iterations` | 最大训练迭代次数 | - |

### Viser专用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--viser_host` | Viser服务器地址 | `0.0.0.0` |
| `--viser_port` | Viser服务器端口 | `8890` |
| `--viser_update_interval` | Viser更新间隔（迭代次数） | `5` |

### 其他参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video` | 是否录制视频 | `False` |
| `--use_proxy` | 是否使用代理 | `False` |
| `--resume` | 是否从检查点恢复 | `False` |
| `--checkpoint` | 检查点文件名 | - |

## 使用示例

### 示例1: 平坦地形训练

```bash
python train_with_viser.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
    --num_envs 4096 \
    --viser_port 8890
```

### 示例2: 粗糙地形训练

```bash
python train_with_viser.py \
    --task Legged-Isaac-Velocity-Rough-Kuavo-S42-v0 \
    --num_envs 4096 \
    --viser_port 8890
```

### 示例3: 调整更新频率（降低开销）

```bash
python train_with_viser.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
    --num_envs 4096 \
    --viser_port 8890 \
    --viser_update_interval 10
```

### 示例4: 使用wandb日志和代理

```bash
python train_with_viser.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
    --num_envs 4096 \
    --viser_port 8890 \
    --logger wandb \
    --log_project_name leju-robot-rl \
    --use_proxy
```

### 示例5: 从检查点恢复训练

```bash
python train_with_viser.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
    --num_envs 4096 \
    --viser_port 8890 \
    --resume=True \
    --checkpoint model_999.pt
```

## 性能优化建议

1. **调整更新间隔**: 如果训练速度较慢，可以增加`--viser_update_interval`的值（如10或20）

2. **减少环境数量**: 如果需要更快的实时反馈，可以减少`--num_envs`的数量

3. **网络访问**: 如果只需在本地查看，设置`--viser_host 127.0.0.1`

4. **端口选择**: 如果端口被占用，使用其他端口（如8891、8892等）

## 界面访问

### 本地访问
```
http://localhost:8890
```

### 远程访问
```
http://<服务器IP>:8890
```

## 技术实现

### 主要组件

1. **ViserTrainingMonitor**: 负责管理viser服务器和UI更新
2. **MonitoredOnPolicyRunner**: 继承自OnPolicyRunner，在训练循环中插入可视化更新
3. **线程安全**: 使用锁和事件机制保证线程安全

### 数据提取

脚本会自动从环境观测中提取：
- 基座速度 (vx, vy, wz)
- 目标速度 (cmd_vx, cmd_vy, cmd_wz)
- 关节位置
- 训练损失

## 故障排除

### 问题1: 端口被占用

**错误信息**:
```
OSError: [Errno 48] Address already in use
```

**解决方法**:
使用不同的端口：
```bash
python train_with_viser.py --viser_port 8891
```

### 问题2: Viser无法访问

**可能原因**:
- 防火墙阻止
- viser服务器未启动
- 端口配置错误

**解决方法**:
1. 检查viser服务器是否启动（查看控制台输出）
2. 尝试使用`127.0.0.1`代替`0.0.0.0`
3. 检查防火墙设置

### 问题3: 性能问题

**可能原因**:
- 更新频率过高
- 环境数量过多
- 网络延迟

**解决方法**:
```bash
# 增加更新间隔
python train_with_viser.py --viser_update_interval 20

# 减少环境数量
python train_with_viser.py --num_envs 2048
```

## 文件结构

```
IsaacLab/
├── scripts/
│   └── rsl_rl/
│       ├── train_with_viser.py      # 带viser监控的训练脚本
│       ├── test_viser.py             # viser功能测试脚本
│       └── VISER_TRAINING.md         # 详细使用文档
└── ...
```

## 系统要求

- Python 3.11+
- Isaac Lab 2.3.0+
- Isaac Sim 5.0+
- viser 1.0.28+
- 不使用headless模式（可以看到3D场景）

## 注意事项

1. **不使用headless模式**: 该脚本使用Isaac Lab的默认渲染，不启用headless模式。

2. **性能影响**: viser可视化会增加一些计算开销，建议调整`--viser_update_interval`来平衡性能和实时性。

3. **网络访问**: 默认绑定到`0.0.0.0`，可以从网络中的其他设备访问。如果只想在本地访问，设置`--viser_host 127.0.0.1`。

4. **端口冲突**: 如果端口被占用，使用`--viser_port`参数指定其他端口。

## 扩展功能

如需添加更多可视化内容，可以修改`ViserTrainingMonitor`类：

```python
# 在_start_server方法中添加新的UI元素
self.custom_text = server.gui.add_text("Custom Info", initial_value="...")

# 在update方法中更新数据
self.custom_text.value = f"...: {custom_value}"

# 在_update_ui方法中更新UI显示
self.custom_text.value = self.custom_data
```

## 相关文档

- [viser官方文档](https://viser.studio/)
- [Isaac Lab文档](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL文档](https://github.com/leggedrobotics/rsl_rl)

## 支持

如有问题，请检查：
1. 控制台输出的错误信息
2. 浏览器开发者工具中的网络请求
3. 防火墙和网络设置

---

**版本**: 1.0
**更新日期**: 2026-05-18
**作者**: AI Assistant