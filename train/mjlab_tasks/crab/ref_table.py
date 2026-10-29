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


def scan_to_heights(frame_z: torch.Tensor, hit_z: torch.Tensor,
                    distances: torch.Tensor, max_distance: float) -> torch.Tensor:
    """把 raycast 输出拼成「躯干高出地形」的高度场，返回 [B, F*N]。

    单独抽成纯函数是为了能被本地自检覆盖 —— 2026-10-29 的教训：
    原实现用 `frame_pos_w[:, :, 2:3]`（[B,1,1]）直接减 `hit_pos_w[..., 2]`
    （[B,N]），广播成 [B,B,N]，展平后长度是 B*N，被下游维度守卫静默丢弃，
    于是 rise_est 全程 0.000、参考步态永远停在平地档，白训 12000 轮。
    B=1 时广播恰好正确，所以回放看不出来，只有多环境才暴露。

    Args:
      frame_z: [B, F, 1] 各帧的世界 z。
      hit_z: [B, F*N] 命中点世界 z（未命中时等于射线原点 z）。
      distances: [B, F*N] 命中距离，-1 表示未命中。
      max_distance: 未命中时填入的高度。
    """
    B, total = int(distances.shape[0]), int(distances.shape[1])
    F = int(frame_z.shape[1])
    N = total // max(F, 1)
    h = frame_z - hit_z.view(B, F, N)                       # [B,F,N]
    miss = (distances < 0).view(B, F, N)
    h = torch.where(miss, torch.full_like(h, float(max_distance)), h)
    return h.reshape(B, total)


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
                 y_min: float = 0.15, y_max: float = 1.00, sigma: float = 0.50):
        """从 height_scan（米，[B, nx*ny]）估计 (rise[B], tread[B], valid[B])。

        两处关键修正（2026-10-29，实跑数据反推）
        ---------------------------------------
        1) 检测带原来写死 |y| ∈ (0.30, 0.90)，但地形中心平台半宽 0.9~1.25 m
           -> 台阶边缘永远在带外 -> rise 全程 0，参考步态永远是平地那一档。
           现在放宽到 (0.15, 1.00)，并把平台宽度改小（见 terrain_cfg）。
        2) tread 原来算的是「最远两条边的跨度 / (边数-1)」。金字塔四周都有台阶，
           ±y 两侧各有一条边，于是 span = 平台宽 + 两级，算出 1.8~2.0 m 这种
           荒谬值。现在只看前进方向 +Y 半侧，取**相邻边的间距均值**。
        """
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
        band = (y_mid.abs() > y_min) & (y_mid.abs() < y_max)
        # rise 两侧都认（机器人可能倒着接近台阶），tread 只看 +Y 半侧
        mask = ok & (dh > edge_thr) & band
        mask_fwd = mask & (y_mid > 0)

        w = torch.exp(-(y_mid ** 2) / (2.0 * sigma ** 2))    # [ny-1]
        den = (w.unsqueeze(0) * mask).sum(-1)
        rise = (w.unsqueeze(0) * dh.clamp(min=0) * mask).sum(-1) / den.clamp_min(1e-6)
        # ⚠️ valid 原来判的是「加权权重和 den > 0.5」。远处的边权重小（|y|=0.85 时
        # e^{-0.85²/2σ²}≈0.24），于是被判无效 -> rise 永远不更新，实测全程 0.000。
        # valid 的语义应该是「有没有找到边」，噪声由 dh > edge_thr 挡。
        valid = mask.any(-1)

        # ---- tread：+Y 半侧相邻边的间距均值 ----
        gap_sum = torch.zeros(B, device=self.device)
        gap_cnt = torch.zeros(B, device=self.device)
        prev_idx = torch.full((B,), -1.0, device=self.device)
        for i in range(ny - 1):
            is_edge = mask_fwd[:, i]
            has_prev = prev_idx >= 0
            gap_sum = torch.where(is_edge & has_prev,
                                  gap_sum + (i - prev_idx) * res, gap_sum)
            gap_cnt = torch.where(is_edge & has_prev, gap_cnt + 1.0, gap_cnt)
            prev_idx = torch.where(is_edge, torch.full_like(prev_idx, float(i)), prev_idx)

        tread = gap_sum / gap_cnt.clamp_min(1.0)
        tread = torch.where(gap_cnt >= 1, tread, torch.full_like(tread, self.mid_tread))
        tread = tread.clamp(float(self.tread[0]), float(self.tread[-1]))
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
    # 4b) 连续两级、踏面 0.3（0.1 m 网格下 0.32 会被量化成 0.3/0.4，故取整值）
    p = np.full(ny, 1.0)
    for t in (0.6, 0.9):
        p[ys >= t] -= 0.12
    cases.append(("stairs tread 0.30", p, 0.12, 0.30))
    # 4c) 远边缘 0.9（旧参数 y_max=0.90 会因严格小于而漏掉，现在应能测到）
    p = np.full(ny, 1.0); p[ys >= 0.9] -= 0.12
    cases.append(("edge at 0.9", p, 0.12, None))

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

    # 5b) 回归：scan_to_heights 在 B>1 时必须给 [B, N]（广播 bug 的守门测试）
    for Bt in (1, 4, 8):
        Fq = 1
        fr = torch.full((Bt, Fq, 3), 1.5)
        fr[..., 2] = 1.5
        d = torch.full((Bt, Fq * nx * ny), 1.0)
        d[:, :nx] = -1.0                                    # 前 nx 条射线 miss
        hitz = torch.full((Bt, Fq * nx * ny), 1.5)
        hitz[:, nx:] = 0.0
        hh = scan_to_heights(fr[:, :, 2:3], hitz, d, 5.0)
        want = (Bt, Fq * nx * ny)
        shape_ok = tuple(hh.shape) == want
        vals_ok = abs(float(hh[0, 0]) - 5.0) < 1e-5 and abs(float(hh[0, nx]) - 1.5) < 1e-5
        print(f"[selftest] scan_to_heights B={Bt} -> {tuple(hh.shape)} "
              f"(want {want}) {'OK' if shape_ok and vals_ok else 'FAIL'}")
        ok_all = ok_all and shape_ok and vals_ok

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
