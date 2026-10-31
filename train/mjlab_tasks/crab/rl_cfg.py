"""PPO runner 配置（与 mjlab 自带 velocity 任务同构，改了个别超参）。"""
from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def _logger_name(default: str = "wandb") -> str:
    """挑一个真能用的 logger。

    ⚠️ 2026-10-29 实跑报错：rsl_rl 的 `WandbLogWriter` 里写死了
    `wandb.Settings(start_method="thread")`，而服务器装的 wandb 已经把
    `start_method` 移除（Settings 变成 pydantic 模型、extra=forbid），于是
    `runner.learn()` 一进来就
        ValidationError: 1 validation error for Settings / start_method
        Extra inputs are not permitted
    修法不是改 rsl_rl（在 venv 里，改了会被覆盖），而是**当场探测**：
    wandb 能接受那次调用就继续用 wandb，否则退回 tensorboard。
    这样换机器/换版本都不会再卡同一个地方。
    """
    if default != "wandb":
        return default
    try:
        import wandb  # noqa: F401
        wandb.Settings(start_method="thread")   # 兼容性探针
        return "wandb"
    except Exception as e:
        print(f"[rl_cfg][warn] wandb 与本机 rsl_rl 不兼容（{type(e).__name__}: "
              f"{str(e).splitlines()[0][:80]}），logger 改用 tensorboard。"
              f" 想强制用 wandb：--agent.logger wandb")
        return "tensorboard"


def elf3_crab_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            # 0.005 -> 0.0015 -> **0.0002**（2026-10-31 第三次调）
            #
            # 前两次都是凭"std 涨得太快"的印象调的，这次算了一下量纲：
            # mjlab 的奖励是 **per-step × dt(0.02)**，所以每步任务奖励只有 ~0.03 量级
            # （实测 2000 轮末：Mean reward -3.37 / ep_len 88.69 = -0.038/步）。
            # 而熵奖励每步 = entropy_coef × H，29 维高斯 σ≈0.87 时 H≈37，
            # 0.0015 时就是 **0.056/步** —— 比任务奖励还大。
            #
            # 后果实测：2000 轮里 action std 0.86 -> 0.90、熵 36.16 -> 37.14 单调上升，
            # ep_len 7.4 s -> 4.4 s -> 1.77 s 单调下降，ref_tracking 0.98 -> 0.09。
            # 优化器在最大化随机性，不是在学走路。
            #
            # 0.0002 时熵项 ≈ 0.0074/步，约为任务奖励的 20%，只做防早熟，不主导。
            # 判据：续训 300 轮内 action std 应从 0.86 **降下来**（< 0.75）。
            #      若仍在涨，说明 0.87 这个初始 std 本身就是问题（init_std=1.0），
            #      下一步把 init_std 降到 0.6 重新开始，而不是继续压 coef。
            entropy_coef=0.0002,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="elf3_crab",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=30_000,
    )
    # 用 setattr 而不是构造参数：服务器这份 mjlab 可能比 main 旧，
    # 万一没有 `logger` 字段也不会 TypeError。
    try:
        cfg.logger = _logger_name(default=getattr(cfg, "logger", "wandb"))
    except Exception:
        pass
    return cfg
