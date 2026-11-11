from pathlib import Path

"""Shared deterministic chunking and noninteractive output test configuration."""

from functools import wraps
import sys

import pytest

import foxes


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


# Keep test chunking deterministic across machines/CI runners.
TEST_CHUNK_SIZE_STATES = 64
TEST_CHUNK_SIZE_POINTS = 500


@pytest.fixture(autouse=True)
def _set_default_engine_chunk_sizes_for_tests(monkeypatch):
    """Make implicit engine chunk sizes independent of the test machine."""
    original_new = foxes.Engine.new.__func__

    def _new_with_test_defaults(cls, engine_type, *args, **kwargs):
        """Create an engine with fixed limits unless the test supplied its own."""
        kwargs.setdefault("chunk_size_states", TEST_CHUNK_SIZE_STATES)
        kwargs.setdefault("chunk_size_points", TEST_CHUNK_SIZE_POINTS)
        return original_new(cls, engine_type, *args, **kwargs)

    monkeypatch.setattr(foxes.Engine, "new", classmethod(_new_with_test_defaults))


@pytest.fixture(autouse=True)
def _disable_output_display_for_tests():
    """Keep all test outputs noninteractive and reject attempts to display."""
    original_init = foxes.ModelBook.__init__
    original_show = foxes.core.Output._show_figures

    @wraps(original_init)
    def initialize_book(self, *args, **kwargs):
        """Create a model book with noninteractive presets and factory options."""
        original_init(self, *args, **kwargs)
        for output in self.outputs.values():
            output.show = False
        for factory in self.outputs.factories:
            factory.kwargs["show"] = False

    def show_without_display(self):
        """Reject enabled display before calling the ordinary output hook."""
        assert self.show is False, "Tests must configure outputs with show=False"
        original_show(self)

    with pytest.MonkeyPatch.context() as policy:
        policy.setattr(foxes.ModelBook, "__init__", initialize_book)
        policy.setattr(foxes.core.Output, "_show_figures", show_without_display)
        yield
