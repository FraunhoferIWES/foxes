"""Example CLI selection and real collector/writer execution."""

import argparse
from pathlib import Path
import runpy
from shutil import which
import sys

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
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


@pytest.mark.parametrize(
    "options", [["-a"], ["--animation"], ["-A", "ani.mp4"], ["--ani_file", "ani.gif"]]
)
@pytest.mark.parametrize(
    "example", ["dyn_wakes", "timelines", "sequential", "sequential_yawcontroller"]
)
def test_examples_reject_removed_animation_flags(monkeypatch, options, example):
    # Animations are selected exclusively through named output models.
    with pytest.raises(SystemExit) as error:
        _parse_example(monkeypatch, EXAMPLES / example / "run.py", options)
    assert error.value.code == 2


@pytest.mark.parametrize("output", ["create_turbine_gif", "create_turbine_mp4"])
def test_dyn_wakes_animation_configuration(monkeypatch, output):
    # Stop at algorithm construction to inspect local model-book customization.
    class Captured(Exception):
        pass

    books = []

    def capture(*args, **kwargs):
        books.append(kwargs["mbook"])
        raise Captured

    script = EXAMPLES / "dyn_wakes/run.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "-nf", "-F", "6", "-R", "1200", "1800", "-o", output],
    )
    monkeypatch.setattr(foxes.algorithms, "Iterative", capture)
    with pytest.raises(Captured):
        runpy.run_path(str(script), run_name="__main__")
    model = books[0].outputs[output]
    assert isinstance(model, foxes.models.outputs.FlowTurbineAnimation)
    assert model.turbines == [1, 4, 7]
    assert model.turbine_variable == FV.REWS
    assert model.ref_xy == [1200, 1800]
    assert model.fps == 6
    assert model.figsize == (5.2, 7)
    assert model.grid_pars == {"resolution": 30, "xmax": 5000, "ymax": 5000}
    assert model.plot_pars == {
        "vmin": 0,
        "quiver_pars": {"angles": "xy", "scale_units": "xy", "scale": 0.013},
        "quiver_n": 35,
        "title": None,
        "rotor_color": "red",
    }
    assert model.layout_pars == {
        "title": "",
        "annotate": 1,
        "anno_delx": -120,
        "anno_dely": -60,
        "alpha": 0,
    }
    assert model.base_name == ("ani.gif" if output.endswith("gif") else "ani.mp4")
    default = foxes.ModelBook().outputs[output]
    assert isinstance(default, foxes.models.outputs.FlowTurbineAnimation)
    assert default.turbines is None
    for name in ("create_gif", "create_mp4"):
        assert type(books[0].outputs[name]) is foxes.models.outputs.FlowAnimation


@pytest.mark.parametrize("suffix", ["gif", "mp4"])
@pytest.mark.parametrize("collect", [False, True])
@pytest.mark.parametrize("example", ["dyn_wakes", "timelines"])
def test_real_dynamic_animation(
    monkeypatch, tmp_path, capsys, suffix, collect, example
):
    # Run the actual dynamic-background example with bounded synthetic inputs.
    if suffix == "mp4" and which("ffmpeg") is None:
        pytest.skip("MP4 encoding requires ffmpeg")
    source = tmp_path / "states.csv"
    pd.DataFrame(
        {"ws": [7.0, 8.0, 9.0], "wd": [270.0, 275.0, 280.0], "ti": [0.07] * 3},
        index=pd.date_range("2020-01-01", periods=3, freq="min"),
    ).to_csv(source)
    monkeypatch.setitem(config, FC.OUTPUT_DIR, tmp_path)
    engine_new = foxes.Engine.new
    monkeypatch.setattr(
        foxes.Engine,
        "new",
        lambda **kwargs: engine_new(
            "numpy", chunk_size_states=2, chunk_size_points=2000, verbosity=0
        ),
    )
    script = EXAMPLES / example / "run.py"
    outputs = (["farm_results"] if collect else []) + [f"create_turbine_{suffix}"]
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "-nf", "-s", str(source), "-o", *outputs],
    )

    # Explicit file outputs must finish even when no farm collector is selected.
    backend = plt.get_backend()
    plt.switch_backend("Agg")
    try:
        if collect:
            namespace = runpy.run_path(str(script), run_name="__main__")
            assert namespace["farm_results"].sizes[FC.STATE] == 3
        else:
            with pytest.raises(SystemExit) as error:
                runpy.run_path(str(script), run_name="__main__")
            assert error.value.code == 0
        path = tmp_path / f"ani_WS.{suffix}"
        if suffix == "gif":
            with Image.open(path) as gif:
                assert gif.n_frames == 3
                assert gif.size == (520, 700)
        else:
            assert path.read_bytes()[4:8] == b"ftyp"
        if example == "timelines":
            assert f"Writing {path}" in capsys.readouterr().out
        assert not plt.get_fignums()
    finally:
        plt.close("all")
        plt.switch_backend(backend)


@pytest.mark.parametrize("example", ["sequential", "sequential_yawcontroller"])
def test_sequential_output_options(monkeypatch, example):
    args = _parse_example(
        monkeypatch, EXAMPLES / example / "run.py", ["-o", "farm_results", "create_mp4"]
    )
    assert args.outputs == ["farm_results", "create_mp4"]


@pytest.mark.parametrize("example", ["sequential", "sequential_yawcontroller"])
@pytest.mark.parametrize("suffix", ["gif", "mp4"])
def test_sequential_local_animation_configuration(monkeypatch, example, suffix):
    # Inspect the actual local model book before any scientific input is read.
    class Captured(Exception):
        pass

    model_book = foxes.models.ModelBook
    book = model_book()
    default_plot_pars = book.outputs[f"create_{suffix}"].plot_pars

    def capture_curve(*args, **kwargs):
        raise Captured

    script = EXAMPLES / example / "run.py"
    monkeypatch.setattr(foxes.models, "ModelBook", lambda: book)
    monkeypatch.setattr(foxes.models.turbine_types, "PCtFile", capture_curve)
    options = ["-V", FV.TI, "-VMAX", "0.2"] if example == "sequential" else []
    monkeypatch.setattr(
        sys, "argv", [str(script), "-F", "6", *options, "-o", f"create_{suffix}"]
    )
    with pytest.raises(Captured):
        runpy.run_path(str(script), run_name="__main__")

    # Preserve all legacy flow/rotor/layout settings without altering defaults.
    model = book.outputs[f"create_{suffix}"]
    assert type(model) is foxes.models.outputs.FlowAnimation
    assert model.base_name == f"ani.{suffix}"
    assert model.fps == 6
    assert model.figsize == (6.4, 4.8)
    assert model.layout_pars == dict(
        title="", annotate=1, anno_delx=-120, anno_dely=-60, alpha=0, s=10
    )
    expected = dict(vmin=0, levels=None, title=None, rotor_color="red")
    if example == "sequential":
        assert model.variables == [FV.TI]
        assert model.grid_pars == dict(
            resolution=10, xmin=-1000, ymin=-1000, xmax=4000, ymax=3000
        )
        expected.update(
            vmax=0.2, quiver_n=157, quiver_pars=dict(scale=0.016, alpha=0.5)
        )
    else:
        assert model.grid_pars == dict(resolution=10)
        expected.update(quiver_n=33, quiver_pars=dict(scale=0.022, alpha=0.5))
    assert model.plot_pars == expected
    assert default_plot_pars is None


@pytest.mark.parametrize("example", ["sequential", "sequential_yawcontroller"])
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("suffix", ["gif", "mp4"])
def test_real_sequential_example_animation(
    monkeypatch, tmp_path, capsys, example, debug, suffix
):
    if suffix == "mp4" and which("ffmpeg") is None:
        pytest.skip("MP4 encoding requires ffmpeg")
    source = tmp_path / "states.csv"
    data = pd.DataFrame(
        {
            FV.WS: [7.0, 8.0, 9.0],
            FV.WD: [270.0, 275.0, 280.0],
            FV.TI: [0.07] * 3,
            FV.RHO: [1.225] * 3,
        },
        index=pd.date_range("2020-01-01", periods=3, freq="10s"),
    )
    data.to_csv(source)
    monkeypatch.setitem(config, FC.OUTPUT_DIR, tmp_path)
    if example == "sequential":
        timeseries = foxes.input.states.Timeseries
        monkeypatch.setattr(
            foxes.input.states,
            "Timeseries",
            lambda **kwargs: timeseries(
                data_source=data, output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO]
            ),
        )
        extra = ["-S", "3"]
    else:
        extra = ["-s", str(source), "-S", "3"]
    engine_new = foxes.Engine.new
    monkeypatch.setattr(
        foxes.Engine,
        "new",
        lambda **kwargs: engine_new("single", chunk_size_points=5000, verbosity=0),
    )
    script = EXAMPLES / example / "run.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "-nf",
            *extra,
            *(["--debug"] if debug else []),
            "-o",
            *(["farm_results"] if suffix == "gif" else []),
            f"create_{suffix}",
        ],
    )
    backend = plt.get_backend()
    plt.switch_backend("Agg")
    try:
        namespace = runpy.run_path(str(script), run_name="__main__")
        assert namespace["algo"].farm_results.sizes[FC.STATE] == 3
        path = tmp_path / f"ani_WS.{suffix}"
        if suffix == "gif":
            with Image.open(path) as gif:
                assert gif.n_frames == 3
                assert gif.size == (640, 480)
        else:
            assert path.read_bytes()[4:8] == b"ftyp"
        assert f"Writing {path}" in capsys.readouterr().out
        assert not plt.get_fignums()
    finally:
        plt.close("all")
        plt.switch_backend(backend)


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


@pytest.mark.parametrize("suffix", ["gif", "mp4"])
def test_timeseries_registered_turbine_animation(monkeypatch, small_examples, suffix):
    # Select the new global model-book output through the unchanged example CLI.
    if suffix == "mp4" and which("ffmpeg") is None:
        pytest.skip("MP4 encoding requires ffmpeg")
    model_book = foxes.models.ModelBook

    def small_model_book():
        book = model_book()
        animation = book.outputs[f"create_turbine_{suffix}"]
        animation.grid_pars = {"n_img_points": (2, 2)}
        animation.figsize = (3, 4)
        animation.turbines = [0, 1]
        return book

    monkeypatch.setattr(foxes.models, "ModelBook", small_model_book)
    script = EXAMPLES / "timeseries/run.py"
    monkeypatch.setattr(
        sys, "argv", [str(script), "-nf", "-o", f"create_turbine_{suffix}"]
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 0
    path = small_examples / f"ani_WS.{suffix}"
    if suffix == "gif":
        with Image.open(path) as gif:
            assert gif.n_frames == 3
            assert gif.size == (300, 400)
    else:
        assert path.read_bytes()[4:8] == b"ftyp"


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
