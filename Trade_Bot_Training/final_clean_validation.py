from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import joblib
from pathlib import Path

# Make sibling training modules importable both when this file is executed
# normally and when pytest/importlib loads it directly by file path.
TRAINING_DIR = Path(__file__).resolve().parent
SYSTEM_ROOT = TRAINING_DIR.parent

# Direct script execution sets sys.path[0] to Trade_Bot_Training, not the
# repository root.  Both locations are required: sibling modules such as
# config_model live in TRAINING_DIR, while the shared package lives under
# SYSTEM_ROOT/shared.  Bootstrap them before any project imports.
for _path in (SYSTEM_ROOT, TRAINING_DIR):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from config_model import (
    FINAL_BACKTEST_BALANCE,
    FINAL_BACKTEST_END_DATE,
    FINAL_BACKTEST_START_DATE,
    DATASET_METADATA_PATH,
    MODEL_PATH,
    REPORT_DIR,
    SYMBOLS,
)
from shared.tradeai_core.target_definition import TARGET_VERSION


TRADEAI_DIR = SYSTEM_ROOT / "TradeAI"
BROKER_PROFILE_PATH = REPORT_DIR / "broker_profile.json"
STAGE4_REPORT = REPORT_DIR / "stage4_production_validation.json"
STAGE5_REPORT = REPORT_DIR / "stage5_risk_validation.json"
RUNTIME_REPORT = REPORT_DIR / "runtime_backtest_summary.json"
FINAL_REPORT = REPORT_DIR / "final_clean_validation.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Required report missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid JSON report: {path}")
    return payload


def _ensure_payoff_target(env: dict[str, str]) -> None:
    """Relabel only when the stored dataset target contract is stale.

    The feature matrix is preserved; only supervised future-path targets are
    rebuilt. This keeps the final command reproducible after a target/payoff
    architecture change without making the user run a hidden manual step.
    """
    metadata = _load_json(DATASET_METADATA_PATH)
    current = str(metadata.get("target_version") or "")
    if current == TARGET_VERSION:
        print(f"Target contract already current: {TARGET_VERSION}")
        return

    print(f"Target contract upgrade: {current or 'UNKNOWN'} -> {TARGET_VERSION}")
    _run(
        "1/4 RELABEL EXECUTION-AWARE PAYOFF TARGET",
        TRAINING_DIR / "relabel_dataset.py",
        env,
    )
    refreshed = _load_json(DATASET_METADATA_PATH)
    if refreshed.get("target_version") != TARGET_VERSION:
        raise RuntimeError("Dataset relabel completed but target contract did not update")


def _require_exact_deployment_model(artifact: dict) -> dict:
    metadata = artifact.get("training_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("Model artifact missing training_metadata")

    deployment = metadata.get("deployment_validation")
    if not isinstance(deployment, dict) or deployment.get("exact_runtime_model") is not True:
        raise RuntimeError(
            "Stage 5 did not promote the exact validated models; "
            "runtime backtest blocked"
        )

    symbols = set((artifact.get("contract") or {}).get("symbols") or [])
    models = set((artifact.get("models") or {}).keys())
    if not symbols or models != symbols:
        raise RuntimeError("Promoted runtime model symbols do not match contract")

    return deployment


def _stage4_diagnostic(report: dict) -> dict:
    """Summarize Stage 4 without treating its pre-risk money gate as final acceptance.

    Stage 4 discovers/validates the decision policy while still using the production
    settings fallback risk. Stage 5 is the authority that calibrates risk on the
    calibration slice and then applies the final money gates. Therefore a Stage 4
    drawdown failure must not prevent Stage 5 from doing the job it exists to do.
    """
    failed_checks: list[str] = []
    gates = report.get("acceptance_gates", {})
    if isinstance(gates, dict):
        for balance, gate in gates.items():
            if not isinstance(gate, dict):
                continue
            for scenario in ("normal", "stress"):
                checks = gate.get(scenario, {})
                if not isinstance(checks, dict):
                    continue
                for name, passed in checks.items():
                    if passed is not True:
                        failed_checks.append(f"{balance}.{scenario}.{name}")
    return {
        "acceptance_pass": report.get("acceptance_pass") is True,
        "failed_checks": failed_checks,
    }

def _validate_broker_profile() -> dict:
    profile = _load_json(BROKER_PROFILE_PATH)
    symbols = profile.get("symbols")
    if not isinstance(symbols, dict):
        raise RuntimeError("Broker profile has no symbols map")
    missing = [symbol for symbol in SYMBOLS if symbol not in symbols]
    if missing:
        raise RuntimeError(
            "Broker profile missing symbols: " + ", ".join(missing)
        )
    return profile


def _backup_current_artifacts() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = SYSTEM_ROOT / f"_stage5_final_backup_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)

    if MODEL_PATH.exists():
        (backup / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
        shutil.copy2(MODEL_PATH, backup / "artifacts" / "models" / MODEL_PATH.name)

    if DATASET_METADATA_PATH.exists():
        dataset_out = backup / "artifacts" / "datasets"
        dataset_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DATASET_METADATA_PATH, dataset_out / DATASET_METADATA_PATH.name)

    reports_out = backup / "artifacts" / "reports"
    for path in (
        STAGE4_REPORT,
        STAGE5_REPORT,
        RUNTIME_REPORT,
        REPORT_DIR / "stage4_decision_policy.json",
        REPORT_DIR / "stage5_decision_policy.json",
        REPORT_DIR / "stage5_risk_policy.json",
        REPORT_DIR / "runtime_backtest_trades.csv",
        REPORT_DIR / "performance_report.txt",
    ):
        if path.exists():
            reports_out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, reports_out / path.name)
    return backup


def _run(label: str, script: Path, env: dict[str, str]) -> None:
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(SYSTEM_ROOT),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["TRADEAI_TRAINING_CUTOFF_DATE"] = FINAL_BACKTEST_START_DATE

    # Every child training script imports shared.tradeai_core.  When Python is
    # launched with an explicit script path, the repository root is not
    # guaranteed to be on sys.path.  Carry an explicit PYTHONPATH into all
    # subprocesses so relabel/trainer/Stage5/runtime behave identically whether
    # launched from PowerShell, the UI, pytest, or another working directory.
    existing = env.get("PYTHONPATH", "").strip()
    required = [str(SYSTEM_ROOT), str(TRAINING_DIR)]
    if existing:
        required.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(required)
    return env


def run() -> dict:
    if FINAL_BACKTEST_BALANCE <= 0:
        raise RuntimeError("FINAL_BACKTEST_BALANCE must be positive")
    if not FINAL_BACKTEST_START_DATE or not FINAL_BACKTEST_END_DATE:
        raise RuntimeError("Final backtest dates are required")

    profile = _validate_broker_profile()
    backup = _backup_current_artifacts()

    print("TRADEAI FINAL CLEAN VALIDATION")
    print(f"Historical train cutoff : {FINAL_BACKTEST_START_DATE}")
    print(f"Untouched backtest      : {FINAL_BACKTEST_START_DATE} -> {FINAL_BACKTEST_END_DATE}")
    print(f"Starting balance        : ${FINAL_BACKTEST_BALANCE:.2f}")
    print(f"Broker profile          : {profile.get('source', 'UNKNOWN')}")
    print(f"Backup                  : {backup.name}")

    env = _base_env()

    # Step 1 is conditional. If the dataset already carries the current payoff
    # target it is a no-op; otherwise the relabel script is run automatically.
    _ensure_payoff_target(env)

    _run("2/4 TRAIN HISTORICAL MODEL", TRAINING_DIR / "trainer.py", env)

    # Stage 5 now owns BOTH broker-money decision-policy calibration and risk
    # calibration, then promotes those exact model objects. Running the old
    # Stage 4 model-training pass here would duplicate work and re-introduce
    # model identity ambiguity.
    _run(
        "3/4 MONEY-AWARE POLICY + RISK VALIDATION",
        TRAINING_DIR / "stage5_risk_validate.py",
        env,
    )
    stage5 = _load_json(STAGE5_REPORT)
    if stage5.get("acceptance_pass") is not True:
        raise RuntimeError("Stage 5 acceptance failed; final backtest blocked")

    promoted_artifact = joblib.load(MODEL_PATH)
    deployment_validation = _require_exact_deployment_model(promoted_artifact)
    print(
        "Exact-model promotion verified | "
        f"split={deployment_validation.get('split_method', 'UNKNOWN')}"
    )

    backtest_env = env.copy()
    backtest_env["TRADEAI_MODE_OVERRIDE"] = "BACKTEST"
    backtest_env["TRADEAI_START_BALANCE"] = f"{FINAL_BACKTEST_BALANCE:.10g}"
    backtest_env["TRADEAI_BACKTEST_START_DATE"] = FINAL_BACKTEST_START_DATE
    backtest_env["TRADEAI_BACKTEST_END_DATE"] = FINAL_BACKTEST_END_DATE

    _run("4/4 CLEAN RUNTIME BACKTEST", TRADEAI_DIR / "demo_bot.py", backtest_env)
    runtime = _load_json(RUNTIME_REPORT)

    report = {
        "stage": "stage7_profit_expectancy_final_validation_v1",
        "created_utc": _utc_now(),
        "training_cutoff_exclusive": FINAL_BACKTEST_START_DATE,
        "backtest_start": FINAL_BACKTEST_START_DATE,
        "backtest_end": FINAL_BACKTEST_END_DATE,
        "start_balance": float(FINAL_BACKTEST_BALANCE),
        "broker_profile_source": profile.get("source"),
        "broker_profile_captured_at_utc": profile.get("captured_at_utc"),
        "backup_directory": str(backup),
        "target_version": TARGET_VERSION,
        "policy_calibration": "broker_money_normal_and_stress",
        "stage5_acceptance": True,
        "exact_runtime_model_verified": True,
        "deployment_validation": deployment_validation,
        "runtime_backtest": runtime,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINAL_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("FINAL CLEAN BACKTEST COMPLETE")
    print("=" * 80)
    print(f"Report: {FINAL_REPORT}")
    return report


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"\nFINAL VALIDATION BLOCKED: {exc}")
        raise SystemExit(1)