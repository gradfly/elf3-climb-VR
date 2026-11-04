"""蹲下 / 起立参考轨迹的加载与缓存。

只读 `train/assets/ref_tables/squat_ref_v1.npz`（由 `train/assets/build_squat_ref.py`
烘焙），里面已经是「29 个可控关节 + 基线高度 + 躯干朝向」的成品，不需要在训练侧
再解析 38 维 qpos（那一步最容易出现关节错位，见烘焙脚本的说明）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# 模块级缓存：同一进程内多个 env / 多次 build 只读一次盘
_CACHE: dict[tuple[str, str, str], "TrajRef"] = {}


@dataclass
class TrajRef:
    key: str
    q: torch.Tensor        # [T, 29] 关节角（names 顺序）
    z: torch.Tensor        # [T]    基座高度
    up: torch.Tensor       # [T, 3] 本体系「重力反方向」（躯干朝向）
    pitch: torch.Tensor    # [T]    前倾角（度，仅供参考/日志）
    names: list[str]
    dt: float
    meta: dict

    @property
    def T(self) -> int:
        return int(self.q.shape[0])

    @property
    def duration(self) -> float:
        return float(self.T - 1) * self.dt


def load_traj(path: str, key: str, device: torch.device | str = "cpu") -> TrajRef:
    ck = (str(path), str(key), str(device))
    if ck in _CACHE:
        return _CACHE[ck]

    d = np.load(path, allow_pickle=True)
    if f"{key}/q" not in d:
        raise KeyError(f"{path} 里没有 {key}/q（现有键：{sorted(d)[:8]}…）")
    names = [str(s) for s in d[f"{key}/names"]]
    traj = TrajRef(
        key=key,
        q=torch.as_tensor(np.asarray(d[f"{key}/q"], dtype=np.float32), device=device),
        z=torch.as_tensor(np.asarray(d[f"{key}/z"], dtype=np.float32), device=device),
        up=torch.as_tensor(np.asarray(d[f"{key}/up"], dtype=np.float32), device=device),
        pitch=torch.as_tensor(np.asarray(d[f"{key}/pitch"], dtype=np.float32), device=device),
        names=names,
        dt=float(d[f"{key}/dt"]),
        meta=dict(d[f"{key}/meta"].item()) if f"{key}/meta" in d else {},
    )
    _CACHE[ck] = traj
    return traj
