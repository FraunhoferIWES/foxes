from pathlib import Path
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
    original_new = foxes.Engine.new.__func__

    def _new_with_test_defaults(cls, engine_type, *args, **kwargs):
        kwargs.setdefault("chunk_size_states", TEST_CHUNK_SIZE_STATES)
        kwargs.setdefault("chunk_size_points", TEST_CHUNK_SIZE_POINTS)
        return original_new(cls, engine_type, *args, **kwargs)

    monkeypatch.setattr(foxes.Engine, "new", classmethod(_new_with_test_defaults))
