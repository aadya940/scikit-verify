"""Guard against a stale installed copy shadowing this repository.

The imported ``skverify`` must be the package sitting next to this
test tree, wherever the repository is checked out (a developer's
directory, CI's workspace). A snapshot install elsewhere on the path
would silently test old code.
"""

from pathlib import Path

import skverify


def pytest_configure(config):
    expected = Path(__file__).resolve().parent.parent / "skverify"
    actual = Path(skverify.__file__).resolve().parent
    assert actual == expected, (
        f"tests would run against a stale install: {skverify.__file__}; "
        "fix with: pip install -e ."
    )
