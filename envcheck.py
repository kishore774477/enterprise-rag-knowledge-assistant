"""Fail fast with a clear message when the Python environment is broken.

Imported first by app.py, ingest.py and evaluate.py, before any third-party package,
so a mismatched virtual environment gives a one-line fix instead of a long traceback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_VENV = PROJECT_DIR / ".venv"

# Packages with compiled (version-specific) extensions, plus the main libraries.
_REQUIRED = ("numpy", "pydantic_core", "langchain_core", "gradio")

_FIX = (
    "Fix: rebuild the environment from the project folder with\n"
    "    .\\setup.ps1 -Force\n"
    "(or delete the .venv folder and follow the setup steps in README.md)."
)


def _running_outside_project_venv() -> bool:
    try:
        return PROJECT_VENV.exists() and Path(sys.prefix).resolve() != PROJECT_VENV.resolve()
    except OSError:
        return False


def check_environment() -> None:
    if sys.version_info < (3, 10):
        sys.exit(f"Python 3.10 or newer is required (you are running {sys.version.split()[0]}).")

    if _running_outside_project_venv():
        print(
            f"Warning: running with Python from {sys.prefix},\n"
            f"not this project's environment ({PROJECT_VENV}).\n"
            "Activate it first:  .\\.venv\\Scripts\\Activate.ps1\n",
            file=sys.stderr,
        )

    for name in _REQUIRED:
        try:
            __import__(name)
        except ImportError as exc:
            version = f"{sys.version_info.major}.{sys.version_info.minor}"
            root = exc.__cause__ or exc  # numpy wraps the real error in a long advice message
            reason = str(root).strip().splitlines()[0] if str(root).strip() else type(root).__name__
            if isinstance(exc, ModuleNotFoundError) and exc.name == name:
                problem = f"The package '{name}' is not installed in this environment."
            else:
                problem = (
                    f"The package '{name}' is installed but cannot be loaded by Python {version}.\n"
                    "This usually means the .venv folder was created with a different Python\n"
                    "version (for example while another project's environment was active)\n"
                    "and still contains packages built for the old version."
                )
            sys.exit(
                f"\nEnvironment problem: {problem}\n"
                f"Python: {sys.executable}\nDetails: {reason}\n\n{_FIX}\n"
            )


if os.getenv("RAG_SKIP_ENV_CHECK") != "1":
    check_environment()
