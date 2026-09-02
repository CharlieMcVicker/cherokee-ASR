"""
Entry point for Cherokee Syllabary Transcriber desktop application and dev server.
"""

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import uvicorn

from syllabary_transcriber.app import SyllabaryApi, app


def configure_desktop_environment() -> None:
    """Configure environment variables for desktop GUI stability."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def ensure_ui_built(ui_dir: Path, dist_dir: Path) -> None:
    """Build React UI if dist directory does not exist."""
    if not dist_dir.exists() or not (dist_dir / "index.html").exists():
        if ui_dir.exists() and (ui_dir / "package.json").exists():
            print(f"Building React UI in {ui_dir}...")
            subprocess.run(["npm", "run", "build"], cwd=ui_dir, check=True)


def run_dev_server(port: int) -> None:
    """Run uvicorn dev server."""
    print(f"Starting dev server on port {port}...")
    uvicorn.run(
        "syllabary_transcriber.app:app", host="127.0.0.1", port=port, reload=True
    )


def run_webview_app(dist_dir: Path) -> None:
    """Launch pywebview application."""
    import webview  # type: ignore

    port = 8765
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    api = SyllabaryApi()

    window = webview.create_window(
        "Cherokee Syllabary Transcriber",
        url=f"http://127.0.0.1:{port}",
        js_api=api,
        width=960,
        height=720,
    )
    webview.start(debug=True)


def main() -> None:
    configure_desktop_environment()
    parser = argparse.ArgumentParser(description="Cherokee Syllabary Transcriber")
    parser.add_argument(
        "--dev", action="store_true", help="Run uvicorn dev server mode"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for dev server (default: 8000)"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    ui_dir = base_dir / "ui"
    dist_dir = ui_dir / "dist"

    if args.dev:
        run_dev_server(args.port)
    else:
        ensure_ui_built(ui_dir, dist_dir)
        run_webview_app(dist_dir)


if __name__ == "__main__":
    main()
