from pathlib import Path
import os
import pytest

from foxes.utils import load_module


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
RUN_ALL_PATH = EXAMPLES_DIR / "run_all.py"
EXAMPLE_DIRS = sorted(
    path.parent
    for path in EXAMPLES_DIR.glob("**/README.md")
    if path.parent.name != "windio"
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CONDA_BUILD") is not None,
    reason="example subprocesses are not run in conda-forge builds",
)

if not RUN_ALL_PATH.is_file():
    pytestmark = pytest.mark.skip(
        reason="examples/run_all.py is not available in the test environment"
    )


@pytest.mark.parametrize(
    "example_dir",
    EXAMPLE_DIRS,
    ids=lambda path: path.relative_to(Path(__file__).parents[2]).as_posix(),
)
def test_example(example_dir):
    run_all = load_module("run_all", RUN_ALL_PATH)

    assert run_all.run_example(str(example_dir), nofig=True) == 0


if __name__ == "__main__":
    pytest.main([__file__])
