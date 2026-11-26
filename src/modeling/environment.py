"""Small helpers for the Phase 5 Free Edition probe (never raise)."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

# pip distribution name -> what the probe calls it
PROBE_PACKAGES: tuple[str, ...] = (
    "mlflow",
    "scikit-learn",
    "lightgbm",
    "optuna",
    "shap",
    "databricks-feature-engineering",
    "matplotlib",
    "pandas",
    "numpy",
)


def package_versions(packages: Sequence[str] = PROBE_PACKAGES) -> dict[str, str | None]:
    """{distribution: installed version or None}."""
    out: dict[str, str | None] = {}
    for name in packages:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def run_check(name: str, fn) -> dict:
    """Run one probe step; record ok / error instead of stopping the notebook."""
    try:
        detail = fn()
        return {"check": name, "ok": True, "detail": detail}
    except Exception as exc:  # the point of a probe is to see what fails
        return {"check": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"[:500]}
