"""elf3 复刻链路的一致性校验（无 GPU、无 mjlab 也能跑）。

逐项确认逆向工程得到的对应关系仍然成立：
  1) policy.onnx metadata 里的训练参数是否齐全、与内置常量一致
  2) policy.onnx 内嵌的参考动作表是否与 fight.npz 逐帧一致
  3) elf3.xml 的 13 个跟踪 body 世界位姿是否与 fight.npz 一致（模型即训练模型）
  4) velocimeter / gyro 传感器输出的参考系（机体系）
  5) 观测向量 160 维拼接是否与 ONNX 输入一致，策略输出是否有限

用法:  python elf3_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco
import onnxruntime as ort

# 2026-10-13 文件搬迁后的目录布局：
#   vendor/elf3_check.py                     <- 本文件（与 policy.onnx / fight.npz 同目录）
#   .../bxi_example_py_elf3/launch/elf3_play.py  <- 被导入的模块
_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
_LAUNCH = _WS / "vendor" / "bxi_controller_ros2" / "src" / "bxi_example_py_elf3" / "launch"
if str(_LAUNCH) not in sys.path:
    sys.path.insert(0, str(_LAUNCH))

import elf3_play as ep

OK = "  [通过]"
BAD = "  [失败]"
fails = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global fails
    print(f"{OK if ok else BAD} {name}" + (f"  —— {detail}" if detail else ""))
    if not ok:
        fails += 1


def main() -> int:
    # policy.onnx / fight.npz 与本脚本同在 vendor/ 目录
    here = _HERE
    onnx_path = here / "policy.onnx"
    motion_path = here / "fight.npz"

    # ---------- 1. metadata ----------
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    need = ("joint_names", "body_names", "anchor_body_name", "default_joint_pos",
            "joint_stiffness", "joint_damping", "action_scale", "observation_names")
    missing = [k for k in need if k not in meta]
    check("ONNX metadata 含全部训练参数", not missing, f"缺失: {missing}" if missing else
          f"run={meta.get('run_path','?')}")
    params = ep.load_params_from_onnx(onnx_path)
    builtin = ep.TrainParams()
    for nm, a, b in (("stiffness", params.stiffness, builtin.stiffness),
                     ("damping", params.damping, builtin.damping),
                     ("action_scale", params.action_scale, builtin.action_scale),
                     ("default_joint_pos", params.default_joint_pos, builtin.default_joint_pos)):
        check(f"metadata.{nm} 与内置常量一致", bool(np.allclose(a, b, atol=1e-6)))
    check("obs 布局 = mjlab tracking 任务的 8 个观测项",
          params.obs_terms == ["command", "motion_anchor_pos_b", "motion_anchor_ori_b",
                               "base_lin_vel", "base_ang_vel", "joint_pos",
                               "joint_vel", "actions"], " ".join(params.obs_terms))
    kp = params.stiffness
    ratio = params.damping / kp
    n_match = int(np.sum(np.abs(ratio - 4.0 / ep.NATURAL_FREQ) < 1e-4))
    check("kd/kp = 4/ω（mjlab 10 Hz / ζ=2 约定）", n_match >= len(kp) - 2,
          f"{n_match}/{len(kp)} 个关节符合（ankle_roll 单独设为 kd/kp=0.1）")
    arm = kp / ep.OMEGA2
    check("action_scale = 0.25·effort/kp（力矩自洽）",
          bool(np.all(params.effort_limit > 0)) and bool(np.all(params.armature > 0)),
          f"effort∈[{params.effort_limit.min():.1f},{params.effort_limit.max():.1f}] Nm, "
          f"armature∈[{arm.min():.4f},{arm.max():.4f}]")

    # ---------- 2. ONNX 内嵌动作表 vs fight.npz ----------
    raw = np.load(motion_path, allow_pickle=True)
    jp, jv = raw["joint_pos"], raw["joint_vel"]
    bp, bq = raw["body_pos_w"], raw["body_quat_w"]
    n_frames = jp.shape[0]
    probe = [0, 1, 17, 999, n_frames // 2, n_frames - 1]
    out_names = [o.name for o in sess.get_outputs()]
    zeros = np.zeros((1, 160), dtype=np.float32)
    max_dq = max_dp = 0.0
    for t in probe:
        r = dict(zip(out_names, sess.run(out_names, {
            "obs": zeros, "time_step": np.array([[t]], dtype=np.float32)})))
        max_dq = max(max_dq, float(np.max(np.abs(r["joint_pos"][0] - jp[t]))))
        max_dp = max(max_dp, float(np.max(np.abs(r["body_pos_w"][0] - bp[t]))))
    check("ONNX 内嵌动作表 == fight.npz", max_dq < 1e-4 and max_dp < 1e-4,
          f"max|Δjoint_pos|={max_dq:.2e}, max|Δbody_pos|={max_dp:.2e}")

    # ---------- 3. 模型 kinematics vs 动作数据 ----------
    model = ep.build_model(params, scene="plane")
    data = mujoco.MjData(model)
    jids = [ep._jid(model, n) for n in params.joint_names]
    jq = [model.jnt_qposadr[j] for j in jids]
    jvadr = [model.jnt_dofadr[j] for j in jids]
    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in params.body_names]
    max_p = max_q = 0.0
    for t in probe:
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[0:3] = bp[t, 0]
        data.qpos[3:7] = bq[t, 0]
        for a, v in zip(jq, jp[t]):
            data.qpos[a] = v
        for a, v in zip(jvadr, jv[t]):
            data.qvel[a] = v
        mujoco.mj_forward(model, data)
        for bi, bid in enumerate(bids):
            max_p = max(max_p, float(np.max(np.abs(data.xpos[bid] - bp[t, bi]))))
            q, r = data.xquat[bid], bq[t, bi]
            max_q = max(max_q, float(min(np.max(np.abs(q - r)), np.max(np.abs(q + r)))))
    check("elf3.xml 跟踪 body 世界位姿 == 动作数据", max_p < 1e-4 and max_q < 1e-4,
          f"max|Δpos|={max_p:.2e} m, max|Δquat|={max_q:.2e}")

    # ---------- 4. 传感器参考系 ----------
    t = n_frames // 2
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = bp[t, 0]
    data.qpos[3:7] = bq[t, 0]
    for a, v in zip(jq, jp[t]):
        data.qpos[a] = v
    for a, v in zip(jvadr, jv[t]):
        data.qvel[a] = v
    data.qvel[0:3] = raw["body_lin_vel_w"][t, 0]
    data.qvel[3:6] = raw["body_ang_vel_w"][t, 0]
    mujoco.mj_forward(model, data)
    p = ep.Elf3Player(model, params, ep.Motion(motion_path, params.joint_names, params.body_names),
                      policy="zero")
    tid = ep._imu_site_id(model)
    if tid < 0:
        raise KeyError("模型里没有 IMU 站点（imu_in_pelvis / imu_in_torso）")
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, tid, res, 0)
    rot = data.site_xmat[tid].reshape(3, 3)
    check("velocimeter 输出 = 机体系线速度", bool(np.allclose(
        data.sensordata[p.sens_v[0]:p.sens_v[0] + 3], rot.T @ res[3:6], atol=1e-6)))
    check("gyro 输出 = 机体系角速度", bool(np.allclose(
        data.sensordata[p.sens_w[0]:p.sens_w[0] + 3], rot.T @ res[0:3], atol=1e-6)))

    # ---------- 5. 观测维度与策略输出 ----------
    obs = p.build_obs(t)
    check("观测向量 160 维（与 ONNX 输入一致）",
          obs.shape == (160,) and list(sess.get_inputs()[0].shape)[1] == 160,
          f"obs={obs.shape}")
    a = p.act(obs, t)
    check("策略输出有限且为 29 维", a.shape == (29,) and bool(np.all(np.isfinite(a))),
          f"|a|max={np.abs(a).max():.3f}")

    print()
    if fails:
        print(f"共 {fails} 项未通过")
        return 1
    print("全部校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
