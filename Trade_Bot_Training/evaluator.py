# filename: Trade_Bot_Traning/evaluator.py

import joblib

from config_model import MODEL_PATH, SYMBOLS, TIMEFRAME_NAME
from feature_engine import FEATURE_HASH, FEATURE_SCHEMA_VERSION
from shared.tradeai_core.model_contract import validate_model_artifact
from shared.tradeai_core.target_definition import PREDICTION_TYPE, TARGET_VERSION


def evaluate():
    print("\n" + "═" * 80)
    print("🧪 MODEL ARTIFACT CONTRACT VALIDATOR — STAGE 3")
    print("═" * 80)
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact missing: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    validate_model_artifact(
        artifact,
        expected_symbols=SYMBOLS,
        expected_timeframe=TIMEFRAME_NAME,
        expected_target_version=TARGET_VERSION,
    )
    print("✔ Artifact type/version valid")
    print(f"✔ Feature schema : {FEATURE_SCHEMA_VERSION}")
    print(f"✔ Feature hash   : {FEATURE_HASH}")
    print(f"✔ Target version : {TARGET_VERSION}")
    print(f"✔ Prediction type: {PREDICTION_TYPE}")
    print(f"✔ Symbols        : {list(artifact['models'].keys())}")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    evaluate()
