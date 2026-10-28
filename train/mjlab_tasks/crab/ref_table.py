"""参考表运行时：读表、估地形、按相位取 q_ref。

纯 numpy/torch，**不 import mjlab**，训练（mjlab）与回放（crab_replay）共用。

坐标约定
--------
* height_scan 的展平顺序：y 外层、x 内层，索引 = iy*Nx + ix（与 mjlab 一致）。
* 射线从躯干向下打，`height = 躯干 z − 命中 z`，所以**地形升高时 height 变小**。
  因此上升边缘用 `dh = h[iy] - h[iy+1]`（正数）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class RefTable:
    """按 (rise, tread) 参数化的周期参考表。"""

    def __init__(self, path: str | Path, device: str | torch.device = "cpu"):
        z = np.load(str(path), allow_pickle=True)
        self.rise = torch.as_tensor(z["rise"], dtype=torch.float32, device=device)
        self.tread = torch.as_tensor(z["tread"], dtype=torch.float32, device=device)
        self.q = torch.as_tensor(z["q"], dtype=torch.float32, device=device)        # [R,T,P,J]
        self.v_ref = torch.as_tensor(z["v_ref"], dtype=torch.float32, device=device)  # [R,T]
        self.joint_names = [str(x) for x in z["joint_names"]]
        self.period = int(z["period_frames"])
        self.dt = float(z["dt"])

        self.R, self.T, self.P, self.J = self.q.shape
        self.q_flat = self.q.reshape(self.R * self.T, self.P, self.J)
        self.v_flat = self.v_ref.reshape(self.R * self.T)
        self.device = device
        self.mid_tread = float(self.tread[self.T // 2])

    # ------------------------------------------------------------ 选表
    def _rise_index(self, rise: torch.Tensor):
        r = rise.clamp(float(self.rise[0]), float(self.rise[-1]))
        dr = float(self.rise[1] - self.rise[0])
        u = (r - float(self.rise[0])) / dr
        i0 = u.floor().long().clamp(0, self.R - 2)
        f = (u - i0.to(torch.float32)).clamp(0.0, 1.0)
        return i0, f

    def _tread_index(self, tread: torch.Tensor) -> torch.Tensor:
        return (tread.unsqueeze(-1) - self.tread).abs().argmin(-1)

    @torch.no_grad()
    def select(self, rise: torch.Tensor, tread: torch.Tensor, phase: torch.Tensor):
        """返回 (q_ref [B,J], v_ref [B])。phase ∈ [0,1)。"""
        rise = rise.to(self.device, torch.float32)
        tread = tread.to(self.device, torch.float32)
        phase = phase.to(self.device, torch.float32)

        i0, f = self._rise_index(rise)
        i1 = (i0 + 1).clamp(max=self.R - 1)
        j = self._tread_index(tread)

        p = torch.remainder(phase, 1.0) * self.P
        p0 = p.floor().long() % self.P
        p1 = (p0 + 1) % self.P
        fr = (p - p.floor()).unsqueeze(-1)

        def gather(i):
            idx = i * self.T + j
            qq = self.q_flat[idx, p0] * (1 - fr) + self.q_flat[idx, p1] * fr
            return qq, self.v_flat[idx]

        qa, va = gather(i0)
        qb, vb = gather(i1)
        fq = f.unsqueeze(-1)
        return qa * (1 - fq) + qb * fq, va * (1 - f) + vb * f

    @torch.no_grad()
    def frame0(self, rise: float = 0.0, tread: float | None = None) -> torch.Tensor:
        """取某档第 0 帧（用于 reset 成参考站姿）。"""
        r = torch.tensor([rise], device=self.device)
        t = torch.tensor([self.mid_tread if tread is None else tread], device=self.device)
        p = torch.zeros(1, device=self.device)
        q, _ = self.select(r, t, p)
        return q[0]

    # ------------------------------------------------------------ 地形估计
    @torch.no_grad()
    def estimate(self, heights: torch.Tensor, *, nx: int = 13, ny: int = 21,
                 size_y: float = 2.0, res: float = 0.1, edge_thr: float = 0.03,
                 y_min: float = 0.30, y_max: float = 0.90, sigma: float = 0.35):
        """从 height_scan（米，[B, nx*ny]）估计 (rise[B], tread[B], valid[B])。"""
        heights = heights.to(self.device, torch.float32)
        B = heights.shape[0]
        h = heights.view(B, ny, nx)

        mid = nx // 2
        lo, hi = max(0, mid - 1), min(nx, mid + 2)
        prof = h[:, :, lo:hi].median(dim=-1).values          # [B,ny]

        finite = torch.isfinite(prof) & (prof < 4.5)          # 5.0 是 miss
        # 上升边缘：地形越高，height 越小 -> dh = h[iy] - h[iy+1] > 0
        dh = prof[:, :-1] - prof[:, 1:]
        ok = finite[:, :-1] & finite[:, 1:]

        y_mid = (-size_y / 2.0
                 + (torch.arange(ny - 1, device=self.device, dtype=torch.float32) + 0.5) * res)
        mask = ok & (dh > edge_thr) & (y_mid.abs() > y_min) & (y_mid.abs() < y_max)

        w = torch.exp(-(y_mid ** 2) / (2.0 * sigma ** 2))    # [ny-1]
        den = (w.unsqueeze(0) * mask).sum(-1)
        rise = (w.unsqueeze(0) * dh.clamp(min=0) * mask).sum(-1) / den.clamp_min(1e-6)
        valid = den > 0.5

        m = mask.to(torch.float32)
        cnt = m.sum(-1)
        first = m.argmax(-1)
        last = (ny - 2) - m.flip(-1).argmax(-1)
        span = (last - first).to(torch.float32) * res
        tread = span / (cnt - 1).clamp(min=1)
        tread = torch.where(cnt >= 2, tread, torch.full_like(tread, self.mid_tread))
        return rise, tread, valid


# ---------------------------------------------------------------- 自测
def _selftest() -> int:
    here = Path(__file__).resolve().parent
    path = here.parent.parent / "assets/ref_tables/crab_ref_v1.npz"
    if not path.exists():
        print(f"[SKIP] 找不到 {path}")
        return 0
    tb = RefTable(path)
    print(f"[table] R={tb.R} T={tb.T} P={tb.P} J={tb.J} "
          f"rise {float(tb.rise[0]):.2f}..{float(tb.rise[-1]):.2f} "
          f"tread {[round(float(x),2) for x in tb.tread]}")

    nx, ny, res, size_y = 13, 21, 0.1, 2.0
    ys = -size_y / 2 + np.arange(ny) * res          # 每个射线点的 y

    def profile_to_heights(prof_y):
        """把 [ny] 的高度剖面铺到整个网格（同一 y 行的 nx 列相同）。"""
        return torch.tensor(np.repeat(prof_y, nx), dtype=torch.float32).unsqueeze(0)

    cases = []
    # 1) 平地：所有高度 1.0
    cases.append(("flat", np.full(ny, 1.0), 0.0, None))
    # 2) 单级 0.12：y>0.4 之后低 0.12
    p = np.full(ny, 1.0); p[ys >= 0.4] -= 0.12
    cases.append(("single 0.12", p, 0.12, None))
    # 3) 单级 0.22
    p = np.full(ny, 1.0); p[ys >= 0.4] -= 0.22
    cases.append(("single 0.22", p, 0.22, None))
    # 4) 两级 0.12 / 0.30，边缘在 0.4、0.7
    p = np.full(ny, 1.0); p[ys >= 0.4] -= 0.12; p[ys >= 0.7] -= 0.12
    cases.append(("two 0.12 tread 0.30", p, 0.12, 0.30))

    ok_all = True
    for name, prof, want_rise, want_tread in cases:
        hs = profile_to_heights(prof)
        rise, tread, valid = tb.estimate(hs)
        r = float(rise[0]); t = float(tread[0]); v = bool(valid[0])
        msg = f"[selftest] {name:22s} rise={r:.3f} (want {want_rise:.2f}) " \
              f"tread={t:.3f} valid={v}"
        good = (not v and want_rise == 0.0) or (v and abs(r - want_rise) < 0.02)
        if want_tread is not None and v:
            good = good and abs(t - want_tread) < 0.04
        msg += "  OK" if good else "  FAIL"
        print(msg)
        ok_all = ok_all and good

    # 5) 含无效射线（5.0）要被过滤
    p = np.full(ny, 1.0); p[ys >= 0.4] -= 0.12
    hs = profile_to_heights(p)
    hs[:, nx * 3: nx * 3 + nx] = 5.0                 # 一整条 y 切片设为 miss
    rise, _, valid = tb.estimate(hs)
    print(f"[selftest] with miss row       rise={float(rise[0]):.3f} valid={bool(valid[0])}")
    ok_all = ok_all and abs(float(rise[0]) - 0.12) < 0.02

    # 6) select 形状/单调性
    B = 4
    q, v = tb.select(torch.full((B,), 0.12), torch.full((B,), 0.30),
                     torch.linspace(0, 0.99, B))
    print(f"[selftest] select -> q{tuple(q.shape)} v{tuple(v.shape)} "
          f"v_mean={float(v.mean()):.3f}")
    ok_all = ok_all and q.shape == (B, 29) and torch.isfinite(q).all()

    print("[selftest]", "ALL OK" if ok_all else "有失败项")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
