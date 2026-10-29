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
            entropy_coef=0.0015,   # 0.005 时 action std 从 0.57 一路涨到 0.71，
                                   # 探索项压过策略梯度 -> 策略停在"很随机但能走"
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
