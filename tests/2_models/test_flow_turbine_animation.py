"""Flow/turbine animation rendering, selections and worker safety."""

import pickle
from shutil import which

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pytest

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.models.outputs import FlowTurbineAnimation, FarmResultsOutput


@pytest.fixture
def animation_algo():
    # Alternate downwind ordering to expose traces using sorted turbine indices.
    states = foxes.input.states.StatesTable(
        pd.DataFrame(
            {FV.WS: [7.0, 8.0, 9.0, 10.0], FV.WD: [90.0, 270.0, 90.0, 270.0]},
            index=[2, 4, 8, 16],
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for x in (0.0, 600.0, 1200.0):
        farm.add_turbine(
            foxes.Turbine(xy=[x, 0.0], turbine_models=["NREL5MW"]), verbosity=0
        )
    return foxes.algorithms.Downwind(
        farm, states, wake_models=["Jensen_linear_k007"], verbosity=0
    )


@pytest.fixture(autouse=True)
def _headless():
    # Restore the caller's backend and figures after file-output checks.
    backend = plt.get_backend()
    plt.switch_backend("Agg")
    try:
        yield
    finally:
        plt.close("all")
        plt.switch_backend(backend)


class _ReadOnlyAnimation(FlowTurbineAnimation):
    def calculate(self, *args, **kwargs):
        # Chunk processing must not retain payloads or mutate shared configuration.
        snapshot = pickle.dumps(self)
        results = super().calculate(*args, **kwargs)
        assert pickle.dumps(self) == snapshot
        return results


@pytest.mark.parametrize("backend", ["numpy", "threads", "process"])
@pytest.mark.parametrize("suffix", ["gif", "mp4"])
def test_real_two_panel_encoding(tmp_path, animation_algo, backend, suffix):
    # Exercise inherited file naming, chunk assembly and both real encoders.
    if suffix == "mp4" and which("ffmpeg") is None:
        pytest.skip("MP4 encoding requires ffmpeg")
    output = _ReadOnlyAnimation(
        turbines=[2, 0],
        turbine_variable=FV.P,
        base_name=f"flow.{suffix}",
        out_dir=tmp_path,
        figsize=(3, 4),
        grid_pars={"n_img_points": (3, 3)},
        plot_pars={"rotor_color": "red", "quiver_n": 1},
        layout_pars={"annotate": 1, "alpha": 0, "title": ""},
        ref_xy=[300.0, 200.0],
    )
    with foxes.Engine.new(
        backend, n_procs=2, chunk_size_states=2, chunk_size_points=4, verbosity=0
    ):
        assert animation_algo.calc_farm(outputs=output) == {}
    path = tmp_path / f"flow_WS.{suffix}"
    if suffix == "gif":
        with Image.open(path) as gif:
            assert gif.n_frames == 4
            assert gif.size == (300, 400)
    else:
        assert path.read_bytes()[4:8] == b"ftyp"
    assert not plt.get_fignums()


@pytest.mark.parametrize("selection", [None, [1, 3], []])
def test_traces_and_grid_match_farm_order(
    tmp_path, monkeypatch, animation_algo, selection
):
    # Capture selected payloads while retaining real rendering and chunk assembly.
    payloads = []
    render = FlowTurbineAnimation._render

    def capture(self, algo, payload):
        payloads.append(payload)
        render(self, algo, payload)

    monkeypatch.setattr(FlowTurbineAnimation, "_render", capture)
    output = FlowTurbineAnimation(
        states_isel=selection,
        out_dir=tmp_path,
        figsize=(3, 4),
        grid_pars={"n_img_points": (3, 3)},
    )
    with foxes.Engine.new(
        "threads", n_procs=2, chunk_size_states=2, chunk_size_points=4, verbosity=0
    ):
        farm = animation_algo.calc_farm(outputs=[output, FarmResultsOutput()])[
            FC.FARM_RESULTS
        ]
        if selection == []:
            assert payloads == []
            assert not list(tmp_path.iterdir())
            return
        payload = payloads[0]
        expected = animation_algo.calc_points(farm, payload["points"][0])
    indices = slice(None) if selection is None else selection
    np.testing.assert_array_equal(payload["states"], farm[FC.STATE][indices])
    np.testing.assert_allclose(payload["farm"][FV.REWS], farm[FV.REWS][indices])
    np.testing.assert_allclose(payload["values"][FV.WS], expected[FV.WS][indices])


def test_panel_artists_and_cleanup(tmp_path, monkeypatch, animation_algo):
    # Inspect actual animation artists and propagate encoder failures.
    from matplotlib.animation import ArtistAnimation

    def inspect(animation, **kwargs):
        flow, trace, _ = animation._fig.axes
        assert len(animation._framedata) == 4
        assert flow.get_subplotspec().get_gridspec().get_height_ratios() == [3, 1]
        assert trace.get_ylabel() == FV.P
        assert [t.get_text() for t in trace.get_legend().get_texts()] == [
            "Turbine 2",
            "Turbine 0",
        ]
        assert [t.get_text() for t in flow.texts if t.get_text().isdigit()] == [
            "0",
            "1",
            "2",
        ]
        assert any(
            np.array_equal(c.get_offsets(), [[300, 200]]) for c in flow.collections
        )
        assert kwargs["writer"] == "pillow"
        assert kwargs["fps"] == 6
        raise OSError("synthetic encoder failure")

    monkeypatch.setattr(ArtistAnimation, "save", inspect)
    output = FlowTurbineAnimation(
        turbines=[2, 0],
        turbine_variable=FV.P,
        ref_xy=[300, 200],
        layout_pars={"annotate": 1, "alpha": 0, "title": ""},
        out_dir=tmp_path,
        fps=6,
        grid_pars={"n_img_points": (3, 3)},
    )
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        with pytest.raises(OSError, match="synthetic encoder failure"):
            animation_algo.calc_farm(outputs=output)
    assert not plt.get_fignums()


@pytest.mark.parametrize(
    "options,match",
    [
        ({"turbines": []}, "turbine selection"),
        ({"turbines": [-1]}, "turbine selection"),
        ({"turbines": [3]}, "turbine selection"),
        ({"turbines": [True]}, "turbine selection"),
        ({"turbine_variable": ""}, "turbine_variable"),
        ({"ref_xy": [0]}, "ref_xy"),
        ({"ref_xy": [0, np.nan]}, "ref_xy"),
        ({"layout_pars": {"ax": None}}, "layout_pars"),
    ],
)
def test_invalid_panel_configuration(tmp_path, animation_algo, options, match):
    # Reject invalid configuration before scientific preparation.
    output = FlowTurbineAnimation(out_dir=tmp_path, **options)
    assert not output.initialized
    with pytest.raises(ValueError, match=match):
        output.initialize(animation_algo)


def test_missing_trace_variable_fails(tmp_path, animation_algo):
    # Missing farm fields must not produce success-shaped empty animations.
    output = FlowTurbineAnimation(
        turbine_variable="no_such_variable",
        out_dir=tmp_path,
        grid_pars={"n_img_points": (2, 2)},
    )
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        with pytest.raises(
            KeyError, match="Missing turbine variable 'no_such_variable'"
        ):
            animation_algo.calc_farm(outputs=output)
    assert not list(tmp_path.iterdir())
