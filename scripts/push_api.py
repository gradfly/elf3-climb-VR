#!/usr/bin/env python3
"""通过 GitHub Git Data API 推送本地提交。

用途：本机到 github.com:443 的连接经常被代理拦（502 / TLS EOF），但
api.github.com 稳定可达。本脚本用 Git Data API（blobs -> tree -> commit ->
PATCH ref）完成推送，不依赖 git 的 smart HTTP 协议。

用法：
    python scripts/push_api.py                # 推送到 main
    python scripts/push_api.py --dry-run      # 只检查，不写入
    python scripts/push_api.py --branch main

前提：
    - 本机能拿到 GitHub 凭据（git credential fill，或环境变量 GH_TOKEN）
    - 本地分支与远端有共同祖先，且该祖先对象在本地存在
      （远端提交若本地没有，先 git fetch）

行为：
    - 把「远端当前提交 .. 本地 HEAD」的所有改动压成一个提交推上去
    - 推送前校验：新 tree 必须与本地 HEAD tree 逐字节一致，否则中止
    - 删除文件数 > 20 时中止（防止误删整仓，历史上踩过）
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

OWNER, REPO = "gradfly", "elf3-climb-VR"
DEFAULT_BRANCH = "main"
MAX_DELETIONS = 20


def get_token() -> str:
    tok = os.environ.get("GH_TOKEN", "").strip()
    if tok:
        return tok
    p = subprocess.run(
        ["git", "credential", "fill"],
        input=b"protocol=https\nhost=github.com\n\n",
        capture_output=True,
    )
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    sys.exit("[ERR] 拿不到 GitHub 凭据（git credential fill 失败），可设 GH_TOKEN")


def api(method: str, path: str, body=None, tok: str = "", tries: int = 10):
    url = f"https://api.github.com/repos/{OWNER}/{REPO}{path}"
    data = json.dumps(body).encode() if body is not None else None
    last = None
    for i in range(1, tries + 1):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {tok}")
        req.add_header("Accept", "application/vnd.github+json")
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404, 409, 422):
                detail = e.read().decode("utf-8", "replace")[:300]
                raise SystemExit(f"[ERR] {method} {path} -> HTTP {e.code}: {detail}")
            last = e
            print(f"  {method} {path} HTTP {e.code}，重试 {i}/{tries}")
        except Exception as e:  # 代理断流 / 超时
            last = e
            print(f"  {method} {path} {type(e).__name__}，重试 {i}/{tries}")
        time.sleep(min(2 * i, 10))
    raise SystemExit(f"[ERR] {method} {path} 重试耗尽: {last}")


def git(*args: str, inp: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], input=inp, capture_output=True, check=True).stdout


def has_object(rev: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", rev], capture_output=True).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = get_token()
    head = git("rev-parse", "HEAD").decode().strip()
    local_tree = git("rev-parse", f"{head}^{{tree}}").decode().strip()

    ref = api("GET", f"/git/ref/heads/{args.branch}", tok=tok)
    base = ref["object"]["sha"]
    print(f"本地 HEAD : {head[:10]}")
    print(f"远端 base : {base[:10]}")

    if base == head:
        print("已一致，无需推送")
        return 0

    base_info = api("GET", f"/git/commits/{base}", tok=tok)
    base_tree = base_info["tree"]["sha"]

    # 远端 base 可能由 API 创建、本地没有该对象。此时找一个 tree 完全相同的
    # 本地提交来算差异；新提交的父仍指向远端 base。
    if has_object(base):
        diff_base = base
    else:
        diff_base = None
        for line in git("rev-list", "HEAD").decode().split():
            if git("rev-parse", f"{line}^{{tree}}").decode().strip() == base_tree:
                diff_base = line
                break
        if diff_base is None:
            sys.exit(
                "[ERR] 远端 base 在本地不存在，且没有 tree 相同的本地提交可作参照。\n"
                "      先 git fetch origin 再试"
            )
        print(f"[INFO] 远端 base 本地缺失，改用同 tree 的本地提交 {diff_base[:10]} 计算差异")

    # 差异清单
    out = git("diff", "--name-status", diff_base, head).decode().strip()
    if not out:
        print("[!] base..HEAD 无文件差异，但提交不同（可能只差元数据）。仍会推送。")
    entries, deletions = [], 0
    for line in out.splitlines():
        parts = line.split("\t")
        status, path = parts[0], parts[-1]
        if status.startswith("D"):
            entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
            deletions += 1
            print(f"  删除 {path}")
            continue
        mode = git("ls-tree", head, path).decode().split()[0]
        raw = git("show", f"{head}:{path}")
        blob = api(
            "POST",
            "/git/blobs",
            {"content": base64.b64encode(raw).decode(), "encoding": "base64"},
            tok=tok,
        )["sha"]
        entries.append({"path": path, "mode": mode, "type": "blob", "sha": blob})
        print(f"  写入 {path} ({mode})")

    if deletions > MAX_DELETIONS:
        sys.exit(f"[ERR] 删除 {deletions} 个文件，超过安全阈值 {MAX_DELETIONS}，已中止")

    tree = api("POST", "/git/trees", {"base_tree": base_tree, "tree": entries}, tok=tok)["sha"]
    if tree != local_tree:
        sys.exit(
            f"[ERR] 新 tree 与本地不一致：remote {tree[:10]} vs local {local_tree[:10]}\n"
            "      可能有重命名/权限位差异，先人工确认，别强推"
        )
    print(f"tree 校验通过: {tree[:10]}")

    if args.dry_run:
        print("[dry-run] 不写入")
        return 0

    msg = git("log", "-1", "--format=%B", head).decode()
    author = {
        "name": git("log", "-1", "--format=%an", head).decode().strip(),
        "email": git("log", "-1", "--format=%ae", head).decode().strip(),
        "date": git("log", "-1", "--format=%aI", head).decode().strip(),
    }
    committer = {
        "name": git("log", "-1", "--format=%cn", head).decode().strip(),
        "email": git("log", "-1", "--format=%ce", head).decode().strip(),
        "date": git("log", "-1", "--format=%cI", head).decode().strip(),
    }
    new = api(
        "POST",
        "/git/commits",
        {"message": msg, "tree": tree, "parents": [base],
         "author": author, "committer": committer},
        tok=tok,
    )
    print(f"新提交: {new['sha'][:10]}")

    api("PATCH", f"/git/refs/heads/{args.branch}", {"sha": new["sha"], "force": True}, tok=tok)
    print(f"已更新远端 {args.branch} -> {new['sha'][:10]}")

    # 校验远端 tree
    remote_tree = api("GET", f"/git/commits/{new['sha']}", tok=tok)["tree"]["sha"]
    if remote_tree == local_tree:
        print("远端 tree 与本地一致 ✔")
    else:
        print("[WARN] 远端 tree 与本地不一致，请检查")

    if new["sha"] != head:
        print(
            "[NOTE] 远端提交 sha 与本地不同（GitHub 会规整消息末尾换行/时区）。\n"
            "       内容（tree）已校验一致，不影响拉取。\n"
            "       想消除本地 ahead/behind：网络允许时 git fetch && git reset --hard origin/" + args.branch
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
