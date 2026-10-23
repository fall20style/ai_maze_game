"""테스트 런처(`test_launcher.py`) 테스트.

런처는 ``server/`` 바깥에 있으므로 경로로 직접 로드한다. 여기서는 프로세스를
띄우지 않고 **인자 전달·포트 선택·준비 대기**만 검증한다.
"""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "test_launcher.py"


@pytest.fixture(scope="module")
def launcher():
    assert LAUNCHER.is_file(), f"런처가 없다: {LAUNCHER}"
    spec = importlib.util.spec_from_file_location("maze_launcher", LAUNCHER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_client_args_are_forwarded(launcher):
    args = launcher.parse_args(["--brain", "rule", "--size", "9", "--seed", "3"])
    assert args.client_args == ["--brain", "rule", "--size", "9", "--seed", "3"]
    assert args.port == launcher.DEFAULT_PORT
    assert args.no_server is False
    assert args.no_client is False


def test_launcher_flags_are_not_forwarded(launcher):
    args = launcher.parse_args(
        ["--port", "0", "--no-server", "--url", "http://x:1", "--brain", "human"]
    )
    assert args.port == 0
    assert args.no_server is True
    assert args.url == "http://x:1"
    assert args.client_args == ["--brain", "human"]


def test_free_port_is_bindable(launcher):
    port = launcher.free_port()
    assert 0 < port < 65536
    assert launcher.port_is_free(port)

    # 실제로 잡고 있으면 사용 중으로 판정해야 한다
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy = sock.getsockname()[1]
        assert launcher.port_is_free(busy) is False


def test_health_wait_times_out_when_nothing_listens(launcher):
    port = launcher.free_port()  # 아무도 listen 하지 않는 포트
    with pytest.raises(launcher.LauncherError):
        launcher.wait_for_health(f"http://127.0.0.1:{port}", timeout=0.5)


def test_both_no_flags_is_rejected(launcher):
    with pytest.raises(SystemExit):
        launcher.main(["--no-server", "--no-client"])
