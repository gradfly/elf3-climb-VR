#!/usr/bin/env python3
"""serve.py —— 零依赖静态服务器（server.mjs 的 Python 备用版）

为什么需要它
------------
`server.mjs` 用了**顶层 await**（`const httpPort = await listen(...)`），
这要求 Node >= 14.8；`web/README.md` 写的前提是 18+。
服务器上装的老 Node（如 12.x）会直接抛：

    SyntaxError: Unexpected reserved word
        at Loader.moduleStrategy (internal/modules/esm/translators.js:133:18)

而训练用的 `.venv` 里一定有 Python 3.11，所以留一个后备：同样是一个纯静态
服务器，MIME 表、禁缓存策略都跟 server.mjs 对齐，行为一致。

用法
----
    python web/tools/serve.py                     # http://0.0.0.0:8080
    python web/tools/serve.py --port 8000
    python web/tools/serve.py --bind 127.0.0.1    # 只给本机
    python web/tools/serve.py --open-path "/?motion=policy_crab"

和 server.mjs 的差别：只有 HTTP，没有 HTTPS 8443（VR 头显要 https 的话
还是得用 server.mjs 或 nginx 反代）。
"""
from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent          # web/

# MIME 表与 server.mjs 里的 MIME 保持一致。
# 关键三个：.wasm 必须是 application/wasm（否则 WASM 不流式编译），
# .mjs/.js 必须是 text/javascript（否则浏览器拒绝当模块执行），
# .stl 给的是模型网格。
EXTRA_MIME = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".xml": "application/xml; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".wasm": "application/wasm",
    ".map": "application/json; charset=utf-8",
    ".stl": "model/stl",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    # Python 3.9+ 的 mimetypes 不认识 .wasm/.mjs/.stl，必须自己补
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        **EXTRA_MIME,
    }

    def end_headers(self) -> None:
        # 与 server.mjs 一致：开发期禁缓存，否则重新烘焙的 motion_*.bin 会被浏览器
        # 拿旧副本，改了没效果。
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("  %s %s\n" % (self.address_string(), fmt % args))


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="elf3 web 静态服务器（Python 后备版，不需要 Node）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080, help="HTTP 端口（默认 8080）")
    ap.add_argument("--bind", default="0.0.0.0", help="绑定地址（默认 0.0.0.0）")
    ap.add_argument("--root", default=str(WEB), help="站点根目录（默认 web/）")
    ap.add_argument("--open-path", default="/", help="启动后打印的地址后缀，如 /?motion=policy_crab")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    if not (root / "index.html").is_file():
        print(f"[错误] {root} 下没有 index.html，--root 指错了？", file=sys.stderr)
        return 2

    handler = functools.partial(Handler, directory=str(root))
    try:
        httpd = Server((a.bind, a.port), handler)
    except OSError as e:
        print(f"[错误] 端口 {a.port} 起不来：{e}\n"
              f"       换一个：--port 8000", file=sys.stderr)
        return 2

    host = "localhost" if a.bind in ("0.0.0.0", "::") else a.bind
    suffix = a.open_path if a.open_path.startswith("/") else "/" + a.open_path
    print(f"站点根目录 : {root}")
    print(f"监听       : {a.bind}:{a.port}")
    print(f"打开       : http://{host}:{a.port}{suffix}")
    print("Ctrl+C 停止", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[停止]")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
