"""Every documented example runs once with bounded inputs and no display."""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import foxes
from foxes.config import config
from foxes.utils import load_module


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
RUNNER = load_module("run_all", EXAMPLES / "run_all.py")


@pytest.mark.parametrize("example", RUNNER.example_dirs(), ids=lambda path: path.name)
def test_example(example, monkeypatch, capsys):
    """Run real scripts/YAML and assert source/process state remains untouched."""
    before = {path.relative_to(example) for path in example.rglob("*")}
    previous_dir, previous_config = Path.cwd(), dict(config)
    figures = plt.get_fignums()

    def forbidden_show(*args, **kwargs):
        pytest.fail("A figure-less example attempted to show a figure")

    monkeypatch.setattr(plt, "show", forbidden_show)
    original_algorithm = foxes.core.Algorithm.__init__
    original_engine = foxes.core.Engine.__init__

    def quiet_algorithm(self, *args, **kwargs):
        original_algorithm(self, *args, **kwargs)
        assert self.verbosity == 0

    def quiet_engine(self, *args, **kwargs):
        original_engine(self, *args, **kwargs)
        assert self.verbosity == 0

    monkeypatch.setattr(foxes.core.Algorithm, "__init__", quiet_algorithm)
    monkeypatch.setattr(foxes.core.Engine, "__init__", quiet_engine)
    assert RUNNER.run_example(example, nofig=True, quick=True) == 0
    text = capsys.readouterr().out
    expected = (
        f"Example {example.name}: {' '.join(RUNNER.command(example, quick=True))}\n"
    )
    assert text == expected
    assert Path.cwd() == previous_dir
    assert dict(config) == previous_config
    assert plt.get_fignums() == figures
    assert {path.relative_to(example) for path in example.rglob("*")} == before
