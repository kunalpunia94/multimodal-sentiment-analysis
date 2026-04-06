"""Run the complete project pipeline in fixed order.

Order:
1) train.py
2) evaluate.py
3) scripts/final_notebook_report.py

Usage:
    python run_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def run_step(step_name: str, command: list[str], cwd: Path) -> None:
    """Run one pipeline step and stop on failure."""
    print(f"\n=== {step_name} ===")
    print("Command:", " ".join(command))
    start = time.time()

    subprocess.run(command, cwd=str(cwd), check=True)

    elapsed = time.time() - start
    print(f"Completed {step_name} in {elapsed/60:.2f} min")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    python_exec = sys.executable

    print("Project root:", project_root)
    print("Python executable:", python_exec)

    steps = [
        ("Training", [python_exec, "train.py"]),
        ("Evaluation", [python_exec, "evaluate.py"]),
        ("Final Notebook-Style Report", [python_exec, "scripts/final_notebook_report.py"]),
    ]

    total_start = time.time()

    for step_name, cmd in steps:
        run_step(step_name, cmd, project_root)

    total_elapsed = time.time() - total_start
    print("\nPipeline finished successfully.")
    print(f"Total time: {total_elapsed/60:.2f} min")


if __name__ == "__main__":
    main()
