# play_s54.py 调试与配置修复记录

**日期**：2026年5月21日
**目标**：基于已有的 `play.py` 脚本，配置并调试针对 Kuavo-S54 机器人的 Play 脚本 `play_s54.py`，使之能成功加载环境和模型进行推理并导出 ONNX 模型。

以下是调试过程中遇到的主要问题及对应的修复总结：

## 1. 补全配置类与环境注册修复
- **问题**：在 `rsl_rl_ppo_cfg.py` 底部，`KuavoS54RoughEnvCfg_PLAY` 类定义不完整导致语法错误。同时在 `__init__.py` 中，Play 环境（`Legged-Isaac-Velocity-Rough-Kuavo-S54-Play-v0`）错误地绑定了训练的策略配置。
- **解决方案**：
  - 删除了残缺的类定义，正确实现了针对 Play 模式的策略配置 `KuavoS54RoughPPORunnerPlayCfg`。
  - 修改 `/config/S54/__init__.py`，使 `Legged-Isaac-Velocity-Rough-Kuavo-S54-Play-v0` 环境准确指向 `KuavoS54RoughPPORunnerPlayCfg`。

## 2. 编写 `play_s54.py` 脚本
- **过程**：参考 `play.py` 复制了环境模拟与推理循环的代码。
- **修复**：
  - 移除了不存在或不必要的特定项目库以避免依赖报错。
  - 添加了 `torch`、`numpy` 库的导入。

## 3. Play 环境 `EventCfg` 报错修复
- **报错**：`AttributeError: 'EventCfg' object has no attribute 'add_joint_default_pos'`。
- **原因**：Play 环境的 `__post_init__` (`KuavoS54RoughEnvCfg_PLAY`) 尝试为环境修改（或置空）一些原本被期望存在于父类 `EventCfg` 中的事件（例如 `add_joint_default_pos`, `add_base_mass` 等），但是当前的 IsaacLab 中 `EventCfg` 并没有或已注释掉这些属性。
- **解决方案**：在 `/config/S54/rough_env_cfg.py` 中将 `KuavoS54RoughEnvCfg_PLAY` 中多余的属性赋值代码注释掉，仅保留支持的项（如 `reset_robot_joints` 等）。

## 4. `obs_normalizer` 缺失错误
- **报错**：`AttributeError: 'OnPolicyRunner' object has no attribute 'obs_normalizer'`。
- **原因**：`rsl_rl` 各个版本的 API 差异导致。在当前工作区的源码中，归一化器被设计成为 `actor_critic` (即 `policy`) 内部的 `actor_obs_normalizer` 属性，而不是保存在的 `OnPolicyRunner` 身上。
- **解决方案**：在模型导出（JIT / ONNX）环节，将获取正常化器的代码从 `ppo_runner.obs_normalizer` 替换为了 `getattr(ppo_runner.alg.policy, "actor_obs_normalizer", None)`。

## 5. `get_observations()` 解包错误
- **报错**：`ValueError: too many values to unpack (expected 2)`。
- **原因**：脚本试图 `obs, _ = env.get_observations()`，但 `VecEnvWrapper` 包装后的 `get_observations()` 接口直接返回了单一的观测字典/张量对象。 
- **解决方案**：将解包改为 `obs = env.get_observations()` 单变量接收。

---

通过以上修复，整个环境加载、模型初始化以及推理过程已完成闭环，支持 KuavoS54 模型的渲染检查与部署格式的导出。