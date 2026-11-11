"""Repository-wide docstring-placement checks for maintained Python sources."""

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _comments_before_docstrings(source):
    """Report same-indentation comments preceding a module/class/function docstring."""
    lines = source.splitlines()
    cleaned = "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line for line in lines
    )
    violations = []
    for node in ast.walk(ast.parse(cleaned)):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) or not ast.get_docstring(node):
            continue
        docstring = node.body[0]
        previous = docstring.lineno - 2
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous < 0:
            continue
        line = lines[previous]
        if (
            line.lstrip().startswith("#")
            and len(line) - len(line.lstrip()) == docstring.col_offset
        ):
            violations.append(docstring.lineno)
    return violations


@pytest.mark.parametrize(
    "source,expected",
    [
        ('# misplaced\n"""Module."""\n', [2]),
        ('def run():\n    # misplaced\n    """Run."""\n', [3]),
        ('class Model:\n    # misplaced\n\n    """Model."""\n', [4]),
        (
            'def run():\n    """Run."""\n    # Explain a non-obvious operation.\n    return 1\n',
            [],
        ),
        ('# Describe the group.\ndef run():\n    """Run."""\n    return 1\n', []),
        ('%matplotlib inline\n"""Notebook."""\n', []),
    ],
)
def test_docstring_comment_detection(source, expected):
    assert _comments_before_docstrings(source) == expected


@pytest.mark.parametrize("folder", ["foxes", "examples", "tests", "notebooks"])
def test_docstrings_precede_implementation_comments(folder):
    failures = []
    extension = "*.ipynb" if folder == "notebooks" else "*.py"
    for path in sorted((ROOT / folder).rglob(extension)):
        if ".ipynb_checkpoints" in path.parts:
            continue
        if extension == "*.py":
            sources = [(str(path.relative_to(ROOT)), path.read_text())]
        else:
            notebook = json.loads(path.read_text())
            sources = [
                (f"{path.relative_to(ROOT)}:cell{index}", "".join(cell["source"]))
                for index, cell in enumerate(notebook["cells"])
                if cell["cell_type"] == "code"
            ]
        for label, source in sources:
            failures.extend(
                f"{label}:{line}" for line in _comments_before_docstrings(source)
            )
    assert not failures, "\n".join(failures)
