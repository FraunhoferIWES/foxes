"""CLI and harness contracts without rerunning the example calculations."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import xarray as xr

import foxes
from foxes.config import config
from foxes.utils import load_module


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
RUNNER = load_module("example_runner_contract", EXAMPLES / "run_all.py")
SCRIPTS = sorted(EXAMPLES.glob("*/run.py"))
POSITIONALS = {
    "icon_dream": ["synthetic.nc", "farms.csv", "curves"],
    "random_timeseries": ["2", "3"],
}
ANIMATIONS = {"timelines", "sequential", "sequential_yawcontroller"}


class ParserCaptured(Exception):
    """Stop after the actual parser and before any model or file operations."""


def parse_example(monkeypatch, script, options):
    """Capture parsed arguments from the real callable entry point."""
    original = argparse.ArgumentParser.parse_args
    captured = []

    def capture(parser, *args, **kwargs):
        captured.append(original(parser, *args, **kwargs))
        raise ParserCaptured

    module = RUNNER._load(script)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(ParserCaptured):
        module.main([*POSITIONALS.get(script.parent.name, []), *options])
    return captured[0]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.parent.name)
def test_default_outputs(monkeypatch, script):
    expected = ["farm_results"]
    if script.parent.name in ANIMATIONS:
        expected.append("create_turbine_gif")
    elif script.parent.name == "wake_fields":
        expected = ["wake_field"]
    elif script.parent.name == "field_data_nc":
        expected = ["farm_results", "slices"]
    args = parse_example(monkeypatch, script, [])
    assert args.outputs == expected
    assert args.verbosity == 2


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.parent.name)
@pytest.mark.parametrize("flag", ["-o", "--outputs"])
def test_output_selection(monkeypatch, script, flag):
    outputs = ["farm_results", "farm_results_write_n2"]
    args = parse_example(monkeypatch, script, [flag, *outputs, "--nofig"])
    assert args.outputs == outputs
    assert args.nofig is True


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.parent.name)
@pytest.mark.parametrize(
    "options",
    [["-v", "0"], ["--verbosity", "0"], ["-v0"], ["-v=0"], ["--verbosity=0"]],
)
def test_explicit_verbosity(monkeypatch, script, options):
    assert parse_example(monkeypatch, script, options).verbosity == 0


@pytest.mark.parametrize("flag", ["-v", "--verbosity"])
def test_runner_verbosity_default_and_override(flag):
    assert RUNNER.parse_args([]).verbosity == 0
    assert RUNNER.parse_args([flag, "3"]).verbosity == 3


@pytest.mark.parametrize(
    "explicit",
    [[], ["-v", "2"], ["--verbosity", "2"], ["-v2"], ["-v=2"], ["--verbosity=2"]],
)
@pytest.mark.parametrize("level", [0, 3])
def test_runner_propagates_verbosity_without_overriding_explicit_choice(
    monkeypatch, tmp_path, explicit, level
):
    from types import SimpleNamespace

    options = []
    monkeypatch.setattr(
        RUNNER, "command", lambda *args: ["python", "run.py", *explicit]
    )
    monkeypatch.setattr(
        RUNNER, "_load", lambda path: SimpleNamespace(main=options.extend)
    )
    RUNNER.run_example(tmp_path, nofig=True, verbosity=level)
    expected = [*explicit, "--nofig"] if explicit else ["--nofig", "-v", str(level)]
    assert options == expected


@pytest.mark.parametrize(
    "example",
    [
        "compare_rotors_pwakes",
        "compare_wakes",
        "gaussian_lookup_partial_wakes",
        "induction",
        "single_state",
        "yawed_wake",
    ],
)
def test_plot_variable_remains_selectable_with_verbosity(monkeypatch, example):
    args = parse_example(
        monkeypatch, EXAMPLES / example / "run.py", ["--var", "CT", "-v", "0"]
    )
    assert args.var == "CT"
    assert args.verbosity == 0


@pytest.mark.parametrize(
    "example", ["dyn_wakes", "timelines", "sequential", "sequential_yawcontroller"]
)
@pytest.mark.parametrize("output_name", ["create_mp4", "create_turbine_mp4"])
@pytest.mark.parametrize("verbosity", [None, 0, 2, 3])
def test_animation_diagnostics_use_script_verbosity(
    monkeypatch, example, output_name, verbosity
):
    # Inspect the real configured output before any scientific inputs are loaded.
    mbook = foxes.models.ModelBook()
    monkeypatch.setattr(foxes.models, "ModelBook", lambda: mbook)

    def stop(*args, **kwargs):
        raise RuntimeError("model book captured")

    monkeypatch.setattr(foxes.models.turbine_types, "PCtFile", stop)
    options = ["-o", output_name, "--nofig"]
    if verbosity is not None:
        options.extend(["-v", str(verbosity)])
    with pytest.raises(RuntimeError, match="model book captured"):
        RUNNER._load(EXAMPLES / example / "run.py").main(options)
    output = mbook.outputs[output_name]
    assert output.verbosity == (2 if verbosity is None else verbosity)
    assert not output.show


@pytest.mark.parametrize("kind", ["yaml", "windio"])
@pytest.mark.parametrize("level", [0, 3])
@pytest.mark.parametrize(
    "explicit", [[], ["--verbosity", "0"], ["--verbosity=2"], ["-v", "2"]]
)
def test_yaml_runner_preserves_explicit_verbosity(
    monkeypatch, tmp_path, kind, level, explicit
):
    import importlib

    calls = []
    module = importlib.import_module(
        f"foxes.input.yaml.{kind}.{kind}"
        if kind == "windio"
        else "foxes.input.yaml.yaml"
    )
    monkeypatch.setattr(
        module, f"foxes_{kind}", lambda *args, **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        RUNNER,
        "command",
        lambda *args: [f"foxes_{kind}", "inputs.yaml", *explicit],
    )
    RUNNER.run_example(tmp_path, nofig=True, verbosity=level)
    expected = level if not explicit else int(explicit[-1].split("=")[-1])
    assert calls == [{"nofig": True, "verbosity": expected}]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.parent.name)
def test_empty_output_rejected(monkeypatch, script):
    with pytest.raises(SystemExit) as error:
        parse_example(monkeypatch, script, ["-o"])
    assert error.value.code == 2


def test_each_example_has_one_command():
    directories = RUNNER.example_dirs()
    assert len(directories) == len(set(directories))
    assert {script.parent for script in SCRIPTS} <= set(directories)
    assert EXAMPLES / "windio" in directories
    for path in directories:
        for quick in (False, True):
            assert RUNNER.command(path, quick)


@pytest.mark.parametrize("content", ["", "<!-- run-all: python run.py -->\n" * 2])
def test_ambiguous_commands_rejected(tmp_path, content):
    (tmp_path / "README.md").write_text(content)
    with pytest.raises(ValueError, match="Expected exactly one"):
        RUNNER.command(tmp_path)


def test_one_dispatch_per_example(monkeypatch):
    calls = []
    monkeypatch.setattr(RUNNER, "run_example", lambda *args: calls.append(args[0]))
    RUNNER.main(["--quick", "--nofig"])
    assert calls == RUNNER.example_dirs()


@pytest.mark.parametrize("failure", [False, True])
def test_isolation_cleanup(tmp_path, failure):
    example = tmp_path / "example"
    example.mkdir()
    (example / "input.txt").write_text("unchanged")
    previous_dir, previous_config = Path.cwd(), dict(config)
    figures = plt.get_fignums()
    work = None

    def execute():
        nonlocal work
        with RUNNER.isolated_example(example, nofig=True) as work:
            assert work != example
            (work / "output.txt").write_text("temporary")
            plt.figure()
            if failure:
                raise RuntimeError("expected failure")

    if failure:
        with pytest.raises(RuntimeError, match="expected failure"):
            execute()
    else:
        execute()
    assert work is not None and not work.exists()
    assert Path.cwd() == previous_dir
    assert dict(config) == previous_config
    assert plt.get_fignums() == figures
    assert [path.name for path in example.iterdir()] == ["input.txt"]


def test_isolation_preserves_utm_zone(tmp_path):
    previous, attributes = dict(config), dict(vars(config))
    try:
        config.set_utm_zone(32, "U")
        with RUNNER.isolated_example(tmp_path, nofig=True):
            assert not config.utm_zone_set
            config.set_utm_zone(33, "U")
        assert config.utm_zone == (32, "U")
    finally:
        dict.clear(config)
        dict.update(config, previous)
        vars(config).clear()
        vars(config).update(attributes)


def test_isolation_preserves_existing_figures(tmp_path):
    baseline = set(plt.get_fignums())
    figure = plt.figure()
    try:
        with RUNNER.isolated_example(tmp_path, nofig=True):
            plt.figure()
        assert set(plt.get_fignums()) == baseline | {figure.number}
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "example,calculations", [("quickstart", 1), ("timeseries", 1), ("power_mask", 2)]
)
@pytest.mark.parametrize("collect", [False, True])
def test_missing_report_fields_do_not_skip_calculations(
    monkeypatch, capsys, example, calculations, collect
):
    calls = []

    def calculate(self, *, outputs):
        calls.append(outputs)
        return xr.Dataset({"P": (("state", "turbine"), [[100.0]])}) if collect else {}

    monkeypatch.setattr(foxes.algorithms.Downwind, "calc_farm", calculate)
    directory = EXAMPLES / example
    with RUNNER.isolated_example(directory, nofig=True) as work:
        RUNNER._load(EXAMPLES / "_quick_inputs.py").prepare_inputs(work, example)
        options = RUNNER.command(directory, quick=True)[2:]
        output = "farm_results_P" if collect else "farm_results_write_n2"
        RUNNER._load(work / "run.py").main(
            [*options, "-o", output, "--nofig", "-v", "1"]
        )
    assert len(calls) == calculations
    text = capsys.readouterr().out
    if example == "quickstart" and collect:
        assert text == "Running example: quickstart\n"
    if example != "quickstart" or not collect:
        assert "Skipping" in text


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.parent.name)
@pytest.mark.parametrize("verbosity", [0, 1, 2, 3])
@pytest.mark.parametrize("delivery", ["direct", "mapping", "absent", "nonfarm"])
def test_farm_dataset_reporting_does_not_depend_on_report_fields(
    monkeypatch, capsys, script, verbosity, delivery
):
    # Stub science, but execute each script's real selection and reporting paths.
    data = xr.Dataset({"P": (("state", "turbine"), [[100.0]])})
    calls = []
    original_calculate = foxes.algorithms.Downwind.calc_farm

    def calculate(self, *args, **kwargs):
        if not isinstance(kwargs.get("outputs"), list):
            return original_calculate(self, *args, **kwargs)
        calls.append(kwargs["outputs"])
        if delivery == "absent":
            return {}
        if delivery == "mapping":
            return {"farm_results": data, "other": "finished"}
        return data

    monkeypatch.setattr(foxes.algorithms.Downwind, "calc_farm", calculate)
    monkeypatch.setattr(foxes.algorithms.Iterative, "calc_farm", calculate)
    sequential = script.parent.name in {"sequential", "sequential_yawcontroller"}
    if sequential:
        monkeypatch.setattr(
            foxes.algorithms.Sequential, "__iter__", lambda self: iter(())
        )
        monkeypatch.setattr(
            foxes.algorithms.Sequential, "farm_results", property(lambda self: data)
        )
    selected = delivery in {"direct", "mapping"}
    outputs = ["farm_results" if sequential else "farm_results_P"]
    if delivery == "mapping":
        outputs.append("create_gif" if sequential else "farm_results_write_n2")
    elif delivery == "absent":
        outputs = ["create_gif" if sequential else "farm_results_write_n2"]
    elif delivery == "nonfarm":
        outputs = ["create_gif" if sequential else "slice_data_xy_states"]
    with RUNNER.isolated_example(script.parent, nofig=True) as work:
        RUNNER._load(EXAMPLES / "_quick_inputs.py").prepare_inputs(
            work, script.parent.name
        )
        options = RUNNER.command(script.parent, quick=True)[2:]
        RUNNER._load(work / "run.py").main(
            [*options, "-o", *outputs, "--nofig", "-v", str(verbosity)]
        )
    text = capsys.readouterr().out
    reports = text.lower().count("farm results:\n")
    count = 1 if sequential else len(calls)
    assert count > 0
    assert reports == (count if selected and verbosity > 1 else 0)
    if selected and verbosity > 1:
        assert text.count(str(data)) == count


@pytest.mark.parametrize("verbosity", [0, 1, 2, 3])
def test_optional_benchmark_reports_each_collected_farm(monkeypatch, capsys, verbosity):
    data = xr.Dataset({"P": (("state", "turbine"), [[100.0]])})
    calls = []

    def calculate(self, *, outputs):
        calls.append(outputs)
        assert self.verbosity == max(verbosity - 1, 0)
        return data

    monkeypatch.setattr(foxes.algorithms.Downwind, "calc_farm", calculate)
    module = RUNNER._load(EXAMPLES / "compare_gaussian_lookup_pwakes" / "run.py")
    with foxes.Engine.new("numpy", verbosity=0) as engine:
        module.benchmark_large_case(
            engine, 1, ["farm_results_P"], nofig=True, verbosity=verbosity
        )
    text = capsys.readouterr().out
    assert calls == [["farm_results_P"]] * 3
    assert text.count(str(data)) == (3 if verbosity > 1 else 0)


@pytest.mark.parametrize("verbosity", [None, 0, 1, 2, 3])
@pytest.mark.parametrize(
    "delivery", ["direct", "nested", "mapping", "nonfarm", "absent", "disabled"]
)
def test_yaml_reports_only_collected_farm_results(
    monkeypatch, capsys, verbosity, delivery
):
    import importlib
    from contextlib import nullcontext
    from types import SimpleNamespace

    module = importlib.import_module("foxes.input.yaml.dict")
    data = xr.Dataset({"P": (("state", "turbine"), [[100.0]])})
    mbook = foxes.ModelBook()
    selector = "farm_results_P"
    supplied = data
    if delivery == "nested":
        selector = foxes.core.OutputList(outputs=[mbook.outputs[selector]])
    elif delivery == "mapping":
        selector = ["farm_results_P", "farm_results_write_n2"]
        supplied = {"farm_results": data, "other": "finished"}
    elif delivery == "nonfarm":
        selector = "slice_data_xy_states"
    elif delivery == "absent":
        selector, supplied = None, {}
    algo = SimpleNamespace(mbook=mbook, calc_farm=lambda **kwargs: supplied)
    monkeypatch.setattr(
        module, "read_dict", lambda *args, **kwargs: (algo, nullcontext())
    )
    monkeypatch.setattr(
        module,
        "run_outputs",
        lambda *args, **kwargs: ([], {}) if kwargs.get("ret_results_storage") else [],
    )
    result = module.run_dict(
        foxes.utils.Dict(
            calc_farm=foxes.utils.Dict(outputs=selector, run=delivery != "disabled")
        ),
        verbosity=verbosity,
    )
    assert result[0] is (None if delivery == "disabled" else supplied)
    text = capsys.readouterr().out
    reports = verbosity is None or verbosity > 1
    collected = delivery in {"direct", "nested", "mapping"}
    assert text.count(str(data)) == int(reports and collected)


def test_yaml_pyplot_is_headless(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("The YAML plotting shortcut attempted display")

    monkeypatch.setattr(plt, "show", forbidden)
    from foxes.input.yaml.dict import run_outputs
    from foxes.utils import Dict

    before = set(plt.get_fignums())
    _, storage = run_outputs(
        Dict(
            outputs=[
                Dict(
                    output_type="plt",
                    functions=[
                        Dict(function="subplots", result_labels=["$fig", "$ax"]),
                        Dict(function="show"),
                    ],
                ),
                Dict(
                    object="$fig",
                    functions=[
                        Dict(function="savefig", fname=tmp_path / "not-created.png")
                    ],
                ),
            ]
        ),
        nofig=True,
        verbosity=0,
    )
    assert storage == {"$fig": None, "$ax": None}
    assert not list(tmp_path.iterdir())
    assert set(plt.get_fignums()) == before


def test_unsupported_example_command_is_explicit(tmp_path):
    (tmp_path / "README.md").write_text("<!-- run-all: unsupported run.py -->\n")
    with pytest.raises(ValueError, match="Unsupported example command"):
        RUNNER.run_example(tmp_path, nofig=True)
