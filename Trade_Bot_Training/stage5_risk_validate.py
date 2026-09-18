"""Compatibility entry point for Stage 11 structural meta-opportunity validation."""
from opportunity_validate import run, _with_exact_deployment_models

__all__ = ["run", "_with_exact_deployment_models"]

if __name__ == "__main__":
    run()
