"""Console entry point for forwarding command-line arguments to Hydra."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _debate_script_path() -> Path:
    script_path = Path(__file__).resolve().with_name("debate.py")
    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find Debate2Create entry script: {script_path}"
        )
    return script_path


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    cmd = [sys.executable, str(_debate_script_path()), *args]
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
