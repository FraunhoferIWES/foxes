import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENS_PATH = REPO_ROOT / "docs/fraunhofer-design/design-tokens.json"
CSS_PATH = REPO_ROOT / "docs/source/_static/iwes-tokens.css"
REFERENCE_PATTERN = re.compile(r"\{([^{}]+)\}")
CSS_VARIABLE_PATTERN = re.compile(r"^\s*(--iwes-[\w-]+):\s*(.+);\s*$", re.MULTILINE)


def _flatten_tokens(node, path=()):
    if "$value" in node:
        return {path: node["$value"]}

    values = {}
    for name, child in node.items():
        if not name.startswith("$"):
            values.update(_flatten_tokens(child, (*path, name)))
    return values


def _resolve_value(value, tokens, resolving=()):
    if not isinstance(value, str):
        return str(value)

    def replace_reference(match):
        path = tuple(match.group(1).split("."))
        assert path in tokens, f"Unknown design-token reference: {'.'.join(path)}"
        assert path not in resolving, f"Circular design-token reference: {path}"
        return _resolve_value(tokens[path], tokens, (*resolving, path))

    return REFERENCE_PATTERN.sub(replace_reference, value)


def _css_variable_name(path):
    parts = []
    for part in path:
        kebab = re.sub(r"(?<!^)(?=[A-Z])", "-", part).lower()
        parts.append(re.sub(r"(?<=[a-z])(?=\d)", "-", kebab))
    return "--iwes-" + "-".join(parts)


def test_sphinx_design_token_adapter_matches_source():
    source = json.loads(TOKENS_PATH.read_text(encoding="utf-8"))
    tokens = _flatten_tokens(source)
    expected = {
        _css_variable_name(path): _resolve_value(value, tokens)
        for path, value in tokens.items()
    }

    declarations = CSS_VARIABLE_PATTERN.findall(CSS_PATH.read_text(encoding="utf-8"))
    actual = dict(declarations)

    assert len(declarations) == len(actual), "CSS adapter contains duplicate variables"
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name] == value, f"Design token drifted: {name}"
