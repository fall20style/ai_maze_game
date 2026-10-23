#!/usr/bin/env python3
"""서버와 SDL 클라이언트를 한 번에 실행하는 테스트 런처.

    python test_launcher.py                          # 서버 + AI 클라이언트
    python test_launcher.py --brain rule             # 규칙(왼손 법칙)
    python test_launcher.py --brain rule --rule frontier
    python test_launcher.py --brain human            # 사람
    python test_launcher.py --size 21 --seed 7       # 나머지 인자는 viewer.app 으로 전달

이미 서버가 떠 있다면:

    python test_launcher.py --no-server --url http://127.0.0.1:8000

서버만 띄우려면:

    python test_launcher.py --no-client

동작
----
1. ``server/`` 에서 uvicorn 을 띄운다 (포트가 사용 중이면 빈 포트로 자동 이동).
2. ``/health`` 가 응답할 때까지 기다린다 (모델 로드는 첫 요청 때 일어나므로 여기선 빠르다).
3. 클라이언트를 실행하고, 창을 닫으면 서버도 함께 종료한다.
4. Ctrl+C 로 언제든 둘 다 정리한다.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
SERVER_DIR = ROOT / "server"
DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"


class LauncherError(RuntimeError):
    """서버를 띄우지 못했거나 준비되지 않았을 때."""


# --------------------------------------------------------------------- 네트워크
def free_port(host: str = DEFAULT_HOST) -> int:
    """OS 가 골라 준 빈 TCP 포트."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def port_is_free(port: int, host: str = DEFAULT_HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _health(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=1.0) as response:
            if response.status != 200:
                return None
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def wait_for_health(
    url: str,
    timeout: float = 60.0,
    process: subprocess.Popen | None = None,
    interval: float = 0.15,
) -> dict:
    """``/health`` 가 200 을 돌려줄 때까지 기다린다.

    ``process`` 가 주어지면 그 프로세스가 먼저 죽었는지도 확인한다
    (포트 충돌·import 오류를 60초 기다리지 않고 바로 알려준다).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise LauncherError(
                f"서버가 시작하자마자 종료했습니다 (exit={process.returncode}). "
                "위쪽 서버 로그를 확인하세요."
            )
        health = _health(url)
        if health is not None:
            return health
        time.sleep(interval)
    raise LauncherError(f"서버가 {timeout:.0f}초 안에 응답하지 않았습니다: {url}")


# --------------------------------------------------------------------- 프로세스
def _env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SERVER_DIR) if not existing else str(SERVER_DIR) + os.pathsep + existing
    )
    # pygame 배너가 로그를 가리지 않게
    env.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    return env


def start_server(host: str, port: int) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.api.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    return subprocess.Popen(cmd, cwd=SERVER_DIR, env=_env())


def start_client(url: str, client_args: Sequence[str]) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "viewer.app", "--server", url, *client_args]
    return subprocess.Popen(cmd, cwd=SERVER_DIR, env=_env())


def stop(process: subprocess.Popen | None, name: str = "프로세스") -> None:
    if process is None or process.poll() is not None:
        return
    print(f"[launcher] {name} 종료 중...", flush=True)
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"[launcher] {name} 를 종료하지 못했습니다 (pid {process.pid})", flush=True)


# ------------------------------------------------------------------------- CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastAPI 서버 + SDL 클라이언트를 함께 실행한다.",
        epilog="알 수 없는 인자는 viewer.app 으로 그대로 전달된다 (예: --brain rule --size 15).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="서버 바인딩 주소")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"서버 포트 (기본 {DEFAULT_PORT}, 0 = 빈 포트 자동)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="이미 실행 중인 서버 주소 (--no-server 와 함께 사용)",
    )
    parser.add_argument(
        "--no-server", action="store_true", help="서버를 띄우지 않고 기존 서버에 연결"
    )
    parser.add_argument(
        "--no-client", action="store_true", help="클라이언트 없이 서버만 실행"
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=60.0,
        help="서버 준비 대기 시간(초)",
    )
    args, extra = parser.parse_known_args(argv)
    args.client_args = list(extra)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.no_server and args.no_client:
        raise SystemExit("--no-server 와 --no-client 는 함께 쓸 수 없습니다")

    if not SERVER_DIR.is_dir():
        raise SystemExit(f"server/ 디렉터리를 찾을 수 없습니다: {SERVER_DIR}")

    server: subprocess.Popen | None = None
    try:
        if args.no_server:
            url = (args.url or f"http://{args.host}:{args.port}").rstrip("/")
            health = wait_for_health(url, args.health_timeout)
            print(f"[launcher] 기존 서버 사용: {url}", flush=True)
        else:
            port = args.port
            if port == 0 or not port_is_free(port, args.host):
                chosen = free_port(args.host)
                if port:
                    print(
                        f"[launcher] 포트 {port} 이(가) 사용 중 → {chosen} 사용",
                        flush=True,
                    )
                port = chosen
            url = f"http://{args.host}:{port}"

            server = start_server(args.host, port)
            print(f"[launcher] 서버 시작: {url} (pid {server.pid})", flush=True)
            health = wait_for_health(url, args.health_timeout, server)

        checkpoint = health.get("checkpoint") if health else None
        print(f"[launcher] 서버 준비 완료 (checkpoint={checkpoint})", flush=True)

        if args.no_client:
            print("[launcher] 서버만 실행 중 — Ctrl+C 로 종료", flush=True)
            while server is not None and server.poll() is None:
                time.sleep(0.3)
            return 0 if server is None else int(server.returncode or 0)

        client = start_client(url, args.client_args)
        print(f"[launcher] 클라이언트 시작 (pid {client.pid})", flush=True)
        return int(client.wait())

    except KeyboardInterrupt:
        print("\n[launcher] 중단 요청", flush=True)
        return 130
    except LauncherError as exc:
        print(f"[launcher] 오류: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        stop(server, "서버")


if __name__ == "__main__":
    raise SystemExit(main())
