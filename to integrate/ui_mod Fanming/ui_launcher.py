"""Launch the Streamlit monitoring UI alongside the existing main loop.

The launcher is responsible for:
* starting the shared DataBridge manager process
* spawning a Streamlit subprocess that renders ``streamlit_ui.py``
* (optionally) spawning ``main.py`` so that gesture control and dashboards run together
* handling shutdown on Ctrl+C and propagating termination to all children
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple

from data_bridge import (
    configure_env_for_bridge,
    find_free_tcp_port,
    start_bridge_server,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动实时监控 UI 与主进程")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="DataBridge 服务监听地址",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="DataBridge 服务端口 (默认自动选择空闲端口)",
    )
    parser.add_argument(
        "--streamlit-port",
        type=int,
        default=8501,
        help="Streamlit Web 界面端口 (默认: 8501)",
    )
    parser.add_argument(
        "--video-source",
        default=0,
        help="模拟视频流使用的 OpenCV source (默认: 0，相机)",
    )
    parser.add_argument(
        "--no-main",
        action="store_true",
        help="仅启动 Streamlit UI，不运行 main.py",
    )
    parser.add_argument(
        "--keep-mock",
        action="store_true",
        help="保持 DataBridge 的模拟数据线程常开 (默认在 main.py 连接后关闭)",
    )
    return parser.parse_args()


def _install_signal_handler(stop_event: threading.Event) -> None:
    def _handle(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def _terminate_processes(processes: List[Tuple[str, subprocess.Popen]]) -> None:
    for name, proc in processes:
        if proc.poll() is None:
            print(f"[Launcher] 结束 {name} (pid={proc.pid})...")
            proc.terminate()
    # second pass with kill if needed
    time.sleep(0.5)
    for name, proc in processes:
        if proc.poll() is None:
            print(f"[Launcher] 强制结束 {name} (pid={proc.pid})")
            proc.kill()
    for _, proc in processes:
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


def main() -> None:
    args = _parse_args()
    stop_event = threading.Event()
    _install_signal_handler(stop_event)

    repo_root = Path(__file__).resolve().parent
    streamlit_entry = repo_root / "streamlit_ui.py"
    main_entry = repo_root / "main.py"

    port = args.port or find_free_tcp_port(args.host)
    manager, bridge, (host, bound_port, authkey) = start_bridge_server(
        host=args.host,
        port=port,
        authkey=None,
        use_mock_generators=True,
        video_source=args.video_source,
    )
    print(f"[Launcher] DataBridge manager @ {host}:{bound_port}")

    common_env = configure_env_for_bridge(os.environ.copy(), host=host, port=bound_port, authkey=authkey)

    processes: List[Tuple[str, subprocess.Popen]] = []
    try:
        streamlit_cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(streamlit_entry),
            "--server.port",
            str(args.streamlit_port),
            "--server.headless",
            "true",
        ]
        print(f"[Launcher] 启动 Streamlit: {' '.join(streamlit_cmd)}")
        streamlit_proc = subprocess.Popen(streamlit_cmd, env=common_env)
        processes.append(("streamlit", streamlit_proc))

        if not args.no_main:
            main_cmd = [sys.executable, str(main_entry)]
            print(f"[Launcher] 启动 main.py: {' '.join(main_cmd)}")
            main_proc = subprocess.Popen(main_cmd, env=common_env)
            processes.append(("main", main_proc))
            if not args.keep_mock:
                # 主进程接管后关闭模拟数据，避免指标重复
                try:
                    bridge.set_mock_mode(False)
                except Exception:
                    pass

        # monitor loop
        while not stop_event.is_set() and processes:
            for name, proc in list(processes):
                ret = proc.poll()
                if ret is not None:
                    print(f"[Launcher] 进程 {name} 退出 (code={ret})")
                    if name == "main" and not args.keep_mock:
                        try:
                            bridge.set_mock_mode(True)
                        except Exception:
                            pass
                    processes.remove((name, proc))
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("[Launcher] 捕获到 Ctrl+C, 正在退出...")
    finally:
        _terminate_processes(processes)
        try:
            bridge.stop()
        except Exception:
            pass
        manager.shutdown()
        print("[Launcher] 清理完成")


if __name__ == "__main__":
    main()
