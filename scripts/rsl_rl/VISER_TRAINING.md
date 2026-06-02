# Viser Training Monitor

使用viser进行实时训练可视化的脚本。

## 功能特性

- **实时训练指标可视化**：
  - 训练迭代次数
  - 平均奖励
  - 平均回合长度
  - 机器人当前速度（vx, vy, wz）
  - 机器人目标速度（cmd_vx, cmd_vy, cmd_wz）

- **训练曲线**：
  - 奖励曲线
  - 回合长度曲线
  - 损失曲线（value_loss, surrogate_loss, policy_loss, entropy_loss）

- **机器人状态**：
  - 关节角度实时显示

## 使用方法

### 基本用法

```bash
# 基础训练（使用默认设置）
python scripts/rsl_rl/train_with_viser.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
    --num_envs 4096

# 指定viser服务器端口
python scripts/rsl_rl/train_with_viser.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
    --num_envs 4096 \
    --viser_port 8890

# 调整更新频率（迭代间隔）
python scripts/rsl_rl/train_with_viser.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
    --num_envs 4096 \
    --viser_update_interval 10
```

### 完整参数

```bash
python scripts/rsl_rl/train_with_viser.py \
    --task <task_name> \
    --num_envs <num_environments> \
    --viser_host <host> \
    --viser_port <port> \
    --viser_update_interval <interval> \
    --video \
    --use_proxy \
    --max_iterations <iterations> \
    --resume \
    --checkpoint <checkpoint_file>
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--viser_host` | Viser服务器地址 | `0.0.0.0` |
| `--viser_port` | Viser服务器端口 | `8890` |
| `--viser_update_interval` | Viser更新间隔（迭代次数） | `5` |
| `--num_envs` | 环境数量 | `4096` |
| `--max_iterations` | 最大训练迭代次数 | `-` |
| `--video` | 是否录制视频 | `False` |
| `--use_proxy` | 是否使用代理 | `False` |
| `--resume` | 是否从检查点恢复 | `False` |
| `--checkpoint` | 检查点文件名 | `-` |

## 访问可视化界面

训练开始后，在浏览器中访问：

```
http://<viser_host>:<viser_port>
```

例如：
```
http://localhost:8890
```

## 界面说明

### Training Status
显示当前训练状态，包括：
- 迭代次数
- 平均奖励
- 平均回合长度
- 当前速度
- 目标速度

### Training Curves
显示训练过程中的曲线：
- Reward: 平均奖励随迭代的变化
- Episode Length: 回合长度随迭代的变化
- Losses: 各种损失随迭代的变化

### Joint States
显示机器人关节角度的实时状态

## 注意事项

1. **不使用headless模式**：该脚本使用Isaac Lab的默认渲染，不启用headless模式，可以看到3D场景。

2. **性能影响**：viser可视化会增加一些计算开销，如果需要最大性能，可以增加`--viser_update_interval`的值。

3. **网络访问**：viser服务器默认绑定到`0.0.0.0`，可以从网络中的其他设备访问。如果只想在本地访问，可以设置为`127.0.0.1`。

4. **端口冲突**：如果端口被占用，可以使用`--viser_port`参数指定其他端口。

## 示例

### 平坦地形训练

```bash
python scripts/rsl_rl/train_with_viser.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
    --num_envs 4096 \
    --viser_port 8890
```

### 粗糙地形训练

```bash
python scripts/rsl_rl/train_with_viser.py \
    --task Legged-Isaac-Velocity-Rough-Kuavo-S42-v0 \
    --num_envs 4096 \
    --viser_port 8890
```

### 使用wandb日志和代理

```bash
python scripts/rsl_rl/train_with_viser.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
    --num_envs 4096 \
    --viser_port 8890 \
    --logger wandb \
    --log_project_name leju-robot-rl \
    --use_proxy
```

### 从检查点恢复训练

```bash
python scripts/rsl_rl/train_with_viser.py \
    --task Legged-Isaac-Velocity-Flat-Kuavo-S42-v0 \
    --num_envs 4096 \
    --viser_port 8890 \
    --resume=True \
    --checkpoint model_999.pt
```

## 技术实现

该脚本通过以下方式实现实时可视化：

1. **Viser服务器**：在后台线程中启动viser web服务器
2. **监控类**：`ViserTrainingMonitor`类负责管理数据和更新UI
3. **线程安全**：使用锁和事件机制保证线程安全
4. **Runner扩展**：`MonitoredOnPolicyRunner`类扩展了标准的`OnPolicyRunner`，在训练循环中插入可视化更新

## 系统要求

- Python 3.11+
- Isaac Lab 2.3.0+
- Isaac Sim 5.0+
- viser包（已通过pip安装）

## 故障排除

### 端口被占用
```
OSError: [Errno 48] Address already in use
```
解决方法：使用不同的端口，例如`--viser_port 8891`

### Viser无法访问
- 检查防火墙设置
- 确认viser服务器正在运行（查看控制台输出）
- 尝试使用`127.0.0.1`代替`0.0.0.0`

### 性能问题
- 增加`--viser_update_interval`的值（如设置为10或20）
- 减少环境数量
- 关闭其他可视化选项