"""Example CLI selection and real collector/writer execution."""

import argparse
from pathlib import Path
import runpy
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
EXCLUDED = {"sequential", "sequential_yawcontroller", "wake_fields"}
SCRIPTS = sorted(
    path for path in EXAMPLES.glob("*/run.py") if path.parent.name not in EXCLUDED
)
POSITIONALS = {
    "icon_dream": ["synthetic_*.nc", "synthetic_farms.csv", "synthetic_curves"],
    "random_timeseries": ["2", "3"],
}


class _ParserCaptured(Exception):
    pass


def _parse_example(monkeypatch, script, options):
    """Run the actual parser and stop before reading any example inputs."""
    # Capture parsed options without constructing models or loading real input files.
    parse_args = argparse.ArgumentParser.parse_args
    captured = []

    def capture(parser, *args, **kwargs):
        captured.append(parse_args(parser, *args, **kwargs))
        raise _ParserCaptured

    argv = [str(script), *POSITIONALS.get(script.parent.name, []), *options]
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        patch.setattr(argparse.ArgumentParser, "parse_args", capture)
        with pytest.raises(_ParserCaptured):
            runpy.run_path(str(script), run_name="__main__")

    return captured[0]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.parent.name)
@pytest.mark.parametrize(
    "options,outputs",
    [
        ([], ["farm_results"]),
        (["-o", "create_gif"], ["create_gif"]),
        (
            ["--outputs", "farm_results", "farm_results_write_n2"],
            ["farm_results", "farm_results_write_n2"],
        ),
    ],
)
def test_example_output_options(monkeypatch, script, options, outputs):
    # Every supported example accepts the same default and short/long selections.
    args = _parse_example(monkeypatch, script, options)
    assert args.outputs == outputs


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.parent.name)
def test_example_output_option_requires_a_name(monkeypatch, script):
    # Reject an incomplete output option before constructing any calculation.
    with pytest.raises(SystemExit) as error:
        _parse_example(monkeypatch, script, ["-o"])
    assert error.value.code == 2


def test_comparison_figure_option_is_separate(monkeypatch):
    # A PNG filename must not consume the common output-model option.
    args = _parse_example(
        monkeypatch,
        EXAMPLES / "compare_gaussian_lookup_pwakes/run.py",
        ["--figure-file", "comparison.png", "-o", "farm_results"],
    )
    assert args.figure_file == Path("comparison.png")
    assert args.outputs == ["farm_results"]


@pytest.fixture
def small_examples(monkeypatch, tmp_path):
    # Bound input size and engine resources while using the real output lifecycle.
    engine_new = foxes.Engine.new

    def single_engine(*args, **kwargs):
        return engine_new(
            "single", chunk_size_states=2, chunk_size_points=8, verbosity=0
        )

    def timeseries(*args, **kwargs):
        return foxes.input.states.StatesTable(
            pd.DataFrame({FV.WS: [7.0, 8.0, 9.0], FV.WD: [270.0, 280.0, 290.0]}),
            output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
            fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
        )

    monkeypatch.setattr(foxes.Engine, "new", single_engine)
    monkeypatch.setattr(foxes.input.states, "Timeseries", timeseries)
    monkeypatch.setitem(config, "output_dir", tmp_path)
    monkeypatch.chdir(tmp_path)

    # Keep comparison figures headless and restore the caller's plotting state.
    backend = plt.get_backend()
    plt.switch_backend("Agg")
    try:
        yield tmp_path
    finally:
        plt.close("all")
        plt.switch_backend(backend)


@pytest.mark.parametrize("example", ["quickstart", "timeseries"])
@pytest.mark.parametrize(
    "outputs",
    [
        ["farm_results"],
        ["farm_results", "farm_results_write_n2"],
        ["farm_results_write_n2"],
    ],
)
def test_real_example_collector_and_writer_returns(
    monkeypatch, small_examples, example, outputs
):
    # Exercise direct Dataset, composed mapping and empty writer-only returns.
    script = EXAMPLES / example / "run.py"
    monkeypatch.setattr(sys, "argv", [str(script), "-nf", "-o", *outputs])
    if "farm_results" in outputs:
        namespace = runpy.run_path(str(script), run_name="__main__")
        farm_results = namespace["farm_results"]
        assert isinstance(farm_results, xr.Dataset)
        assert farm_results.sizes[FC.STATE] == 3
        assert FV.P in farm_results
    else:
        with pytest.raises(SystemExit) as error:
            runpy.run_path(str(script), run_name="__main__")
        assert error.value.code == 0

    # Files must contain every input state, not just exist after a mocked call.
    files = sorted(small_examples.glob("farm_results*.nc"))
    if "farm_results_write_n2" in outputs:
        assert len(files) == 2
        counts = []
        for path in files:
            with xr.open_dataset(path) as dataset:
                counts.append(dataset.sizes[FC.STATE])
                assert FV.P in dataset
        assert counts == [2, 1]
    else:
        assert not files


def test_unknown_output_model_fails_explicitly(monkeypatch, small_examples):
    # Model-book resolution errors must not be mistaken for writer-only success.
    script = EXAMPLES / "quickstart/run.py"
    monkeypatch.setattr(sys, "argv", [str(script), "-o", "no_such_output"])
    with pytest.raises(KeyError, match="no_such_output"):
        runpy.run_path(str(script), run_name="__main__")


@pytest.mark.parametrize("example", ["power_mask", "sector_management"])
@pytest.mark.parametrize("with_writer", [False, True])
def test_example_specific_collector_fields(
    monkeypatch, small_examples, example, with_writer
):
    # Keep custom reporting fields when selecting the local farm-results model.
    if example == "power_mask":
        pd.DataFrame({FV.WS: [7.0, 8.0, 9.0]}).to_csv("states.csv")
        pd.DataFrame({"Pmax_0": [3000.0] * 3, "Pmax_1": [3500.0] * 3}).to_csv(
            "power_mask.csv", index=False
        )
        options = ["-nt", "2"]
    else:
        pd.DataFrame(
            {"ws": [7.0, 8.0, 9.0], "wd": [270.0] * 3, "weight": [1 / 3] * 3}
        ).to_csv("states.csv")
        pd.DataFrame(
            {
                "tname": ["T0"],
                "WD_min": [250.0],
                "WD_max": [290.0],
                "WS_min": [0.0],
                "WS_max": [100.0],
                FV.MAX_P: [3000.0],
            }
        ).to_csv("rules.csv", index=False)
        options = ["-s", "states.csv", "-sm", "rules.csv"]

    # Composing a writer must still expose the dataset used by legacy reporting.
    outputs = ["farm_results"]
    if with_writer:
        outputs.append("farm_results_write_n2")
    script = EXAMPLES / example / "run.py"
    monkeypatch.setattr(sys, "argv", [str(script), "-nf", *options, "-o", *outputs])
    namespace = runpy.run_path(str(script), run_name="__main__")
    farm_results = namespace["farm_results"]
    assert farm_results.sizes[FC.STATE] == 3
    assert (FV.MAX_P in farm_results) == (example == "sector_management")
    assert FV.CT in farm_results
    assert bool(list(small_examples.glob("farm_results*.nc"))) == with_writer


def test_comparison_saves_selected_figure_file(monkeypatch, small_examples):
    # Exercise the renamed PNG flag independently of composed chunk outputs.
    script = EXAMPLES / "compare_gaussian_lookup_pwakes/run.py"
    image = small_examples / "figures/comparison.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "-nf",
            "--y-span",
            "0.1",
            "--step",
            "0.1",
            "--figure-file",
            str(image),
            "-o",
            "farm_results",
            "farm_results_write_n2",
        ],
    )
    runpy.run_path(str(script), run_name="__main__")
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not plt.get_fignums()


@pytest.mark.parametrize(
    "example,options,n_calls",
    [
        (
            "compare_wakes",
            ["-w", "Jensen_linear_k007", "Bastankhah2014_linear_k004"],
            4,
        ),
        (
            "compare_rotors_pwakes",
            [
                "-r",
                "centre",
                "-p",
                "centre",
                "grid4",
                "-y0",
                "-10",
                "-y1",
                "10",
                "-ys",
                "10",
            ],
            2,
        ),
        (
            "compare_rotors_pwakes",
            [
                "-r",
                "centre",
                "grid4",
                "-p",
                "centre",
                "-y0",
                "-10",
                "-y1",
                "10",
                "-ys",
                "10",
            ],
            2,
        ),
        (
            "compare_rotors_pwakes",
            [
                "-r",
                "centre",
                "grid4",
                "-p",
                "centre",
                "grid4",
                "-y0",
                "-10",
                "-y1",
                "10",
                "-ys",
                "10",
            ],
            2,
        ),
        ("compare_gaussian_lookup_pwakes", ["--y-span", "0.1", "--step", "0.1"], 3),
        (
            "compare_gaussian_lookup_pwakes",
            ["--y-span", "0.1", "--step", "0.1", "-rt", "-nt", "2"],
            6,
        ),
        ("power_mask", ["-nt", "2"], 2),
    ],
)
def test_writer_only_comparisons_run_every_case(
    monkeypatch, small_examples, example, options, n_calls
):
    # Supply synthetic power-mask files rather than reading a developer's data.
    if example == "power_mask":
        pd.DataFrame({FV.WS: [7.0, 8.0, 9.0]}).to_csv("states.csv")
        pd.DataFrame({"Pmax_0": [3000.0] * 3, "Pmax_1": [3500.0] * 3}).to_csv(
            "power_mask.csv", index=False
        )

    # Observe the real farm runs, including comparisons after the first empty result.
    calc_farm = foxes.algorithms.Downwind.calc_farm
    calls = []

    def calculate(algo, *args, **kwargs):
        calls.append(kwargs["outputs"])
        return calc_farm(algo, *args, **kwargs)

    monkeypatch.setattr(foxes.algorithms.Downwind, "calc_farm", calculate)
    script = EXAMPLES / example / "run.py"
    outputs = ["farm_results_write_n2"]
    monkeypatch.setattr(sys, "argv", [str(script), "-nf", *options, "-o", *outputs])
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as error:
        assert error.code == 0

    assert calls == [outputs] * n_calls
    assert list(small_examples.glob("farm_results*.nc"))
    assert not plt.get_fignums()
