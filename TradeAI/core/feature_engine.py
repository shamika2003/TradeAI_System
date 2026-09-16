# filename: TradeAI/core/feature_engine.py
# Compatibility wrapper: TradeAI and training now use one shared engine.

from __future__ import annotations

import sys
from pathlib import Path


def _add_system_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "shared" / "tradeai_core").is_dir():
            root = parent
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            return root
    raise RuntimeError(
        "TradeAI_System root not found. Expected shared/tradeai_core beside the project folders."
    )


SYSTEM_ROOT = _add_system_root()

from shared.tradeai_core.feature_engine import FeatureTransformer  # noqa: E402
from shared.tradeai_core.feature_schema import (  # noqa: E402
    FEATURE_HASH,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
)

__all__ = [
    "FeatureTransformer",
    "FEATURE_HASH",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "SYSTEM_ROOT",
]
