"""
Launcher for the Football Performance Intelligence app.

Ensures the demo artefacts exist (database + trained model), then starts
Streamlit. Usage:  python run_app.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _ensure_built():
    from src.ml.train import BUNDLE_PATH
    from src import config
    if not BUNDLE_PATH.exists() or not config.DB_PATH.exists():
        print("First run — building demo artefacts (data + model)…")
        import build_demo
        build_demo.main()


def main():
    sys.path.insert(0, str(ROOT))
    _ensure_built()
    app = ROOT / "app" / "main.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app),
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false"])


if __name__ == "__main__":
    main()
