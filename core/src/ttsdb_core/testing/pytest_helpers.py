"""Pytest hook helpers shared across model packages."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def add_variant_option(parser) -> None:
    """Add --variant CLI option to pytest.

    Call from pytest_addoption in each model's conftest.py:

        def pytest_addoption(parser):
            add_variant_option(parser)

    Then use get_selected_variants() to retrieve the value.
    """
    parser.addoption(
        "--variant",
        action="store",
        default=None,
        help="Run integration tests only for this variant (e.g., --variant=v3). "
        "Can be comma-separated for multiple variants (e.g., --variant=v3,v4). "
        "If not specified, runs all variants.",
    )


def get_selected_variants(config) -> list[str] | None:
    """Get the list of variants selected via --variant, or None for all.

    Returns:
        List of variant names if --variant was specified, None otherwise.
    """
    variant_opt = config.getoption("--variant", default=None)
    if variant_opt is None:
        return None
    # Support comma-separated variants
    return [v.strip() for v in variant_opt.split(",") if v.strip()]


def configure_integration_marker(config, *, extra: str = "") -> None:
    """Register the shared `integration` marker."""

    suffix = f" {extra}".rstrip()
    config.addinivalue_line(
        "markers",
        f"integration: mark test as integration test (requires model weights){suffix}",
    )


def skip_integration_by_default(config, items) -> None:
    """Skip `@pytest.mark.integration` unless `-m integration` is used."""

    # Avoid importing pytest at module import time (keeps this "test-only" helper
    # lightweight and side-effect free outside test runs).
    import pytest

    marker_expr = config.getoption("-m")
    if marker_expr != "integration":
        skip_integration = pytest.mark.skip(reason="integration tests require -m integration")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def write_integration_result(model_root: Path, session, exitstatus: int) -> None:
    """Write integration_result.json when integration tests were run.

    Call from pytest_sessionfinish in each model's conftest. Only writes if
    the session was run with -m integration.
    """
    marker_expr = session.config.getoption("-m", default="")
    if marker_expr != "integration":
        return

    # Check if specific variants were requested
    selected_variants = get_selected_variants(session.config)

    examples_dir = Path(model_root) / "audio_examples"
    variants_tested: list[str] = []
    artifacts: list[str] = []
    if examples_dir.exists():
        for p in sorted(examples_dir.iterdir()):
            if p.is_dir():
                # Only include if this variant was selected (or no filter)
                if selected_variants is None or p.name in selected_variants:
                    variants_tested.append(p.name)
                    for f in sorted(p.glob("*.wav")):
                        artifacts.append(f.relative_to(examples_dir).as_posix())
            else:
                if p.suffix.lower() == ".wav":
                    artifacts.append(p.name)
                    if not variants_tested:
                        variants_tested.append("default")

    result = {
        "model_id": model_root.name,
        "passed": exitstatus == 0,
        "timestamp": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "variants_tested": variants_tested,
        "variants_filter": selected_variants,  # Record what was filtered
        "artifacts_count": len(artifacts),
        "artifacts": artifacts[:50],
    }
    out_path = Path(model_root) / "integration_result.json"
    out_path.write_text(json.dumps(result, indent=2))
