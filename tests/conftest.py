"""Guard against a stale non-editable install shadowing the dev tree."""

import skverify


def pytest_configure(config):
    assert "veripy-dev" in skverify.__file__, (
        f"tests would run against a stale install: {skverify.__file__}; "
        "fix with: pip install -e ."
    )
