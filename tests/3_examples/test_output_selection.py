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
DEFAULT_OUTPUTS = {
    "binned_states": ["rose_plot"],
    "compare_gaussian_lookup_pwakes": ["comparison"],
    "compare_rotors_pwakes": ["comparison"],
    "compare_wakes": ["cross_sections", "centreline"],
    "dyn_wakes": ["create_turbine_gif"],
    "field_data_nc": ["slices", "flow_plot_xy_mean"],
    "gaussian_lookup_partial_wakes": ["flow_plot_xy_states"],
    "induction": ["flow_plot_xy_states"],
    "power_mask": ["state_turbine_map"],
    "random_timeseries": ["farm_layout", "rose_plot"],
    "scan_row": ["state_turbine_map"],
    "sector_management": ["power_rose_0", "power_rose_1"],
    "sequential": ["create_turbine_gif"],
    "sequential_yawcontroller": ["create_turbine_gif"],
    "single_state": ["flow_plot_xy_states", "flow_plot_xz_states"],
    "streamline_wakes": ["flow_plot_xy_states"],
    "tab_file": ["rose_plot"],
    "timelines": ["create_turbine_gif"],
    "timeseries": ["state_turbine_map"],
    "timeseries_slurm": ["state_turbine_map"],
    "wake_fields": ["wake_field"],
    "wind_rose": ["rose_plot"],
    "yawed_wake": ["flow_plot_xy_states", "flow_plot_yz_states"],
}


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
    args = parse_example(monkeypatch, script, [])
    assert args.results == ["farm_results"]
    assert args.outputs == DEFAULT_OUTPUTS.get(
        script.parent.name, ["state_turbine_table"]
    )
    assert args.verbosity == 2


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.parent.name)
@pytest.mark.parametrize("flag", ["-o", "--outputs"])
def test_output_selection(monkeypatch, script, flag):
    outputs = ["farm_results", "farm_results_write_n2"]
    args = parse_example(monkeypatch, script, [flag, *outputs, "--nofig"])
    assert args.outputs == outputs
    assert args.nofig is True


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.parent.name)
@pytest.mark.parametrize("flag", ["-R", "--results"])
def test_result_selection_is_independent(monkeypatch, script, flag):
    results = ["farm_results_P", "total_results_mean"]
    args = parse_example(
        monkeypatch, script, [flag, *results, "-o", "farm_layout", "-nf"]
    )
    assert args.results == results
    assert args.outputs == ["farm_layout"]
    assert args.nofig


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.parent.name)
def test_workflow_remains_monolithic(script):
    import ast

    functions = [
        node.name
        for node in ast.walk(ast.parse(script.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert functions == ["main"]


@pytest.mark.parametrize("example", ["timeseries", "dyn_wakes", "timelines"])
def test_rotor_and_reference_options_remain_independent(monkeypatch, example):
    options = ["-r", "grid4", "-R", "farm_results_P"]
    if example != "timeseries":
        options.extend(["--ref_xy", "100", "200"])
    args = parse_example(monkeypatch, EXAMPLES / example / "run.py", options)
    assert args.rotor == "grid4"
    assert args.results == ["farm_results_P"]
    if example != "timeseries":
        assert args.ref_xy == [100, 200]


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
    assert output.nofig


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
@pytest.mark.parametrize("flag", ["-o", "-R"])
def test_empty_output_rejected(monkeypatch, script, flag):
    with pytest.raises(SystemExit) as error:
        parse_example(monkeypatch, script, [flag])
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


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.parent.name)
@pytest.mark.parametrize("verbosity", [0, 1, 2, 3])
@pytest.mark.parametrize("delivery", ["farm", "grouped", "absent", "nonfarm"])
def test_result_reporting_and_postprocessing_are_independent(
    monkeypatch, capsys, script, verbosity, delivery
):
    # Execute every script's real CLI and both stages without repeating science.
    data = xr.Dataset({"P": (("state", "turbine"), [[100.0]])})
    result_mapping = {
        "farm": {"farm_results": data, "other": "calculation finished"},
        "grouped": {"mean_total_farm_results": data},
        "absent": {},
        "nonfarm": {"probe": data},
    }[delivery]
    calculations, processed = [], []

    def calculate(self, *, outputs, ret_dict):
        assert ret_dict
        assert outputs == ["farm_results_P"]
        calculations.append(outputs)
        return result_mapping

    def process(self, *, outputs, farm_results, ret_dict):
        assert ret_dict
        if isinstance(outputs, list):
            # Sequential collects its accumulated iterator data through this API.
            calculations.append(outputs)
            return result_mapping
        assert outputs == "state_turbine_table"
        assert farm_results is result_mapping.get("farm_results")
        assert self.mbook.outputs[outputs].nofig
        assert calculations
        outputs_marker = f"post-processing result {len(calculations)}"
        processed.append(outputs_marker)
        return {"post": outputs_marker}

    monkeypatch.setattr(foxes.algorithms.Downwind, "calc_farm", calculate)
    monkeypatch.setattr(foxes.algorithms.Iterative, "calc_farm", calculate)
    monkeypatch.setattr(foxes.algorithms.Downwind, "calc_outputs", process)
    sequential = script.parent.name in {"sequential", "sequential_yawcontroller"}
    if sequential:
        monkeypatch.setattr(
            foxes.algorithms.Sequential, "__iter__", lambda self: iter(())
        )
        monkeypatch.setattr(
            foxes.algorithms.Sequential, "farm_results", property(lambda self: data)
        )
    with RUNNER.isolated_example(script.parent, nofig=True) as work:
        RUNNER._load(EXAMPLES / "_quick_inputs.py").prepare_inputs(
            work, script.parent.name
        )
        options = RUNNER.command(script.parent, quick=True)[2:]
        RUNNER._load(work / "run.py").main(
            [
                *options,
                "-R",
                "farm_results_P",
                "-o",
                "state_turbine_table",
                "-nf",
                "-v",
                str(verbosity),
            ]
        )
    text = capsys.readouterr().out
    assert calculations
    assert len(processed) == len(calculations)
    for key, value in result_mapping.items():
        assert text.count(f"\n{key}:\n{value}") == (
            len(calculations) if verbosity > 1 else 0
        )
    for value in processed:
        assert (f"\npost:\n{value}" in text) == (verbosity > 1)
    assert "Skipping" not in text


@pytest.mark.parametrize("verbosity", [0, 1, 2, 3])
def test_optional_benchmark_reports_each_collected_farm(monkeypatch, capsys, verbosity):
    data = xr.Dataset({"P": (("state", "turbine"), [[100.0]])})
    calls = []

    def calculate(self, *, outputs, ret_dict):
        calls.append(outputs)
        assert ret_dict
        assert self.verbosity == max(verbosity - 1, 0)
        return {"farm_results": data}

    monkeypatch.setattr(foxes.algorithms.Downwind, "calc_farm", calculate)
    monkeypatch.setattr(
        foxes.algorithms.Downwind, "calc_outputs", lambda *args, **kwargs: {}
    )
    module = RUNNER._load(EXAMPLES / "compare_gaussian_lookup_pwakes" / "run.py")
    with RUNNER.isolated_example(
        EXAMPLES / "compare_gaussian_lookup_pwakes", nofig=True
    ):
        module.main(
            [
                "-e",
                "numpy",
                "-y",
                "0.1",
                "-s",
                "0.1",
                "--runtime-benchmark",
                "--benchmark-turbines",
                "1",
                "-R",
                "farm_results_P",
                "-nf",
                "-v",
                str(verbosity),
            ]
        )
    text = capsys.readouterr().out
    assert calls == [["farm_results_P"]] * 6
    assert text.count(str(data)) == (6 if verbosity > 1 else 0)


@pytest.mark.parametrize(
    "script",
    [path for path in SCRIPTS if not path.parent.name.startswith("sequential")],
    ids=lambda path: path.parent.name,
)
def test_nofig_is_forwarded_without_filtering_either_selection(monkeypatch, script):
    calls = []
    selected_results = ["farm_results", "create_gif", "flow_plot_xy_states"]
    selected_outputs = ["create_gif", "flow_plot_xy_states"]

    def calculate(self, *, outputs, ret_dict):
        assert outputs == selected_results
        assert all(self.mbook.outputs[name].nofig for name in outputs)
        assert ret_dict
        calls.append(outputs)
        return {"farm_results": xr.Dataset()}

    def process(self, *, outputs, farm_results, ret_dict):
        assert outputs in selected_outputs
        assert self.mbook.outputs[outputs].nofig
        assert ret_dict
        assert isinstance(farm_results, xr.Dataset)
        return {}

    monkeypatch.setattr(foxes.algorithms.Downwind, "calc_farm", calculate)
    monkeypatch.setattr(foxes.algorithms.Iterative, "calc_farm", calculate)
    monkeypatch.setattr(foxes.algorithms.Downwind, "calc_outputs", process)
    with RUNNER.isolated_example(script.parent, nofig=True) as work:
        RUNNER._load(EXAMPLES / "_quick_inputs.py").prepare_inputs(
            work, script.parent.name
        )
        options = RUNNER.command(script.parent, quick=True)[2:]
        RUNNER._load(work / "run.py").main(
            [
                *options,
                "-R",
                "farm_results",
                "create_gif",
                "flow_plot_xy_states",
                "-o",
                "create_gif",
                "flow_plot_xy_states",
                "-nf",
                "-v",
                "0",
            ]
        )
    assert calls


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
