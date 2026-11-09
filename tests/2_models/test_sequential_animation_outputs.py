"""Model-book animation adapters for state-by-state execution."""

from shutil import which

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pytest

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.algorithms.iterative.models.convergence import ConvVarDelta
from foxes.models.outputs import FlowAnimation


def _algorithm(plugin):
    states = foxes.input.states.StatesTable(
        pd.DataFrame(
            {FV.WS: [7.0, 8.0, 9.0], FV.WD: [270.0, 275.0, 280.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="min"),
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.07, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for x in (0.0, 600.0):
        farm.add_turbine(
            foxes.Turbine(xy=[x, 0.0], turbine_models=["NREL5MW"]), verbosity=0
        )
    return foxes.algorithms.Sequential(
        farm,
        states,
        plugins=[plugin],
        wake_models=["Jensen_linear_k007"],
        wake_frame="seq_dyn_wakes_1min",
        conv_crit=ConvVarDelta({FV.REWS: 1e-6}),
        verbosity=0,
    )


@pytest.fixture(autouse=True)
def _headless():
    backend = plt.get_backend()
    plt.switch_backend("Agg")
    try:
        yield
    finally:
        plt.close("all")
        plt.switch_backend(backend)


@pytest.mark.parametrize("suffix", ["gif", "mp4"])
@pytest.mark.parametrize("debug", [False, True])
def test_real_sequential_animation(tmp_path, suffix, debug):
    if suffix == "mp4" and which("ffmpeg") is None:
        pytest.skip("MP4 encoding requires ffmpeg")
    plugin = foxes.output.SeqAnimationOutputPlugin([f"create_{suffix}"], debug=debug)
    algo = _algorithm(plugin)
    output = algo.mbook.outputs[f"create_{suffix}"]
    output.out_dir = tmp_path
    output.grid_pars = {"n_img_points": (3, 3)}
    output.figsize = (3, 3)
    with foxes.Engine.new("single", chunk_size_points=4, verbosity=0):
        for _ in algo:
            assert not list(tmp_path.iterdir())
    path = tmp_path / f"ani_WS.{suffix}"
    if suffix == "gif":
        with Image.open(path) as gif:
            assert gif.n_frames == 3
            assert gif.size == (300, 300)
    else:
        assert path.read_bytes()[4:8] == b"ftyp"
    assert not plt.get_fignums()
    assert all(not chunks for _, _, _, chunks in plugin._runs)


@pytest.mark.parametrize("selection", [None, [1], []])
def test_sequential_selection_and_numerical_values(tmp_path, monkeypatch, selection):
    output = FlowAnimation(
        out_dir=tmp_path,
        states_isel=selection,
        grid_pars={"n_img_points": (2, 2)},
    )
    plugin = foxes.output.SeqAnimationOutputPlugin([output])
    algo = _algorithm(plugin)
    payloads, expected = [], []
    monkeypatch.setattr(
        output, "_render", lambda algo, payload, *args: payloads.append(payload)
    )
    with foxes.Engine.new("single", chunk_size_points=2, verbosity=0):
        for fres in algo:
            points = plugin._runs[0][1]
            expected.append(algo.calc_points(fres, points)[FV.WS].to_numpy())
    if selection == []:
        assert payloads == []
    else:
        indices = slice(None) if selection is None else selection
        payload = payloads[0]
        np.testing.assert_array_equal(
            payload["states"], algo.farm_results[FC.STATE][indices]
        )
        np.testing.assert_allclose(
            payload["values"][FV.WS], np.concatenate(expected)[indices]
        )


@pytest.mark.parametrize(
    "outputs,error,match",
    [
        (["unknown"], KeyError, "unknown"),
        (["farm_results"], TypeError, "FlowAnimation"),
    ],
)
def test_invalid_sequential_animation_selection(outputs, error, match):
    algo = _algorithm(foxes.output.SeqAnimationOutputPlugin(outputs))
    with foxes.Engine.new("single", verbosity=0):
        with pytest.raises(error, match=match):
            list(algo)


@pytest.mark.parametrize("chunk", [0, (1, 0), [0, 2]])
def test_sequential_step_chunk_selection(tmp_path, monkeypatch, chunk):
    output = FlowAnimation(
        out_dir=tmp_path, chunk=chunk, grid_pars={"n_img_points": (2, 2)}
    )
    plugin = foxes.output.SeqAnimationOutputPlugin([output])
    algo = _algorithm(plugin)
    payloads = []
    monkeypatch.setattr(
        output, "_render", lambda a, payload, *args: payloads.append(payload)
    )
    with foxes.Engine.new("single", chunk_size_points=2, verbosity=0):
        list(algo)
    expected = (
        [0, 2]
        if isinstance(chunk, list)
        else [chunk[0] if isinstance(chunk, tuple) else chunk]
    )
    np.testing.assert_array_equal(
        payloads[0]["states"], algo.farm_results[FC.STATE][expected]
    )


@pytest.mark.parametrize("chunk", [3, (0, 1)])
def test_unavailable_sequential_chunk(tmp_path, chunk):
    output = FlowAnimation(
        out_dir=tmp_path, chunk=chunk, grid_pars={"n_img_points": (2, 2)}
    )
    algo = _algorithm(foxes.output.SeqAnimationOutputPlugin([output]))
    with foxes.Engine.new("single", verbosity=0):
        with pytest.raises(ValueError, match="unavailable"):
            list(algo)


def test_sequential_turbine_and_multiple_formats(tmp_path):
    outputs = [
        foxes.models.outputs.FlowTurbineAnimation(
            base_name=f"traces{i}.gif",
            out_dir=tmp_path,
            turbines=[1],
            fps=5 / 3,
            grid_pars={"n_img_points": (2, 2)},
            layout_pars={"annotate": 1, "alpha": 0, "title": ""},
        )
        for i in range(2)
    ]
    algo = _algorithm(foxes.output.SeqAnimationOutputPlugin(outputs, debug=True))
    with foxes.Engine.new("single", chunk_size_points=2, verbosity=0):
        list(algo)
    for i in range(2):
        with Image.open(tmp_path / f"traces{i}_WS.gif") as gif:
            assert gif.n_frames == 3
            assert gif.info["duration"] == 600


def test_no_animation_does_not_collect_debug():
    plugin = foxes.output.SeqAnimationOutputPlugin([], debug=True)
    with foxes.Engine.new("single", verbosity=0):
        list(_algorithm(plugin))
    assert plugin._debug is None
