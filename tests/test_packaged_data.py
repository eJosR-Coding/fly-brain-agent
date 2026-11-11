"""Packaged data files must be committed: a fresh clone otherwise has no presets or glossary."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = sorted((ROOT / "src/flybrain/data").glob("*.json"))


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="not a git checkout")
@pytest.mark.parametrize("path", PACKAGED, ids=lambda p: p.name)
def test_packaged_json_is_tracked_by_git(path):
    rel = path.relative_to(ROOT).as_posix()
    ignored = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT).returncode == 0
    assert not ignored, f"{rel} is gitignored, so a fresh clone would not have it"


def test_the_package_ships_presets_and_glossary():
    assert {p.name for p in PACKAGED} >= {"presets.json", "glossary.json"}
