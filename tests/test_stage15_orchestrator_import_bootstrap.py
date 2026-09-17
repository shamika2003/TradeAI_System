from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Trade_Bot_Training"
FINAL = TRAINING / "final_clean_validation.py"


def test_final_validation_bootstraps_repo_root_when_loaded_as_script_path():
    # Simulate the user's exact direct-script launch, but do not execute run().
    # runpy with a non-__main__ name executes all imports and module setup.
    code = (
        "import runpy; "
        f"m=runpy.run_path({str(FINAL)!r}, run_name='tradeai_bootstrap_test'); "
        "assert m['SYSTEM_ROOT'].resolve() == __import__('pathlib').Path(" + repr(str(ROOT)) + ").resolve()"
    )
    env = os.environ.copy()
    # Deliberately remove project-specific PYTHONPATH so the module has to
    # bootstrap itself instead of passing accidentally because pytest did it.
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT.parent),
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_child_environment_contains_repo_root_and_training_dir():
    import importlib.util

    spec = importlib.util.spec_from_file_location("tradeai_final_bootstrap_env", FINAL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    env = module._base_env()
    entries = [Path(p).resolve() for p in env["PYTHONPATH"].split(os.pathsep) if p]
    assert ROOT.resolve() in entries
    assert TRAINING.resolve() in entries
