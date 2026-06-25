from pathlib import Path
import subprocess


def test():
    thisdir = Path(__file__).resolve().parent
    example_dir = thisdir.parent.parent / "examples" / "sequential"

    result = subprocess.run(
        ["uv", "run", "run.py", "-nf"],
        cwd=example_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise AssertionError(
            f"Sequential example failed with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


if __name__ == "__main__":
    test()
