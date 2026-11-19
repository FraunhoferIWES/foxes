"""Flow animation collection, bounded point evaluation and file assembly."""

from copy import deepcopy
import pickle
from shutil import which
from threading import get_ident

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData, MData, TData
from foxes.models.outputs import FlowAnimation, FlowTurbineAnimation, FarmResultsOutput
from foxes.utils import Dict


@pytest.fixture(scope="module", autouse=True)
def _headless_rendering():
    # File-output tests must not create GUI resources collected by worker threads.
    import matplotlib.pyplot as plt

    backend = plt.get_backend()
    plt.switch_backend("Agg")
    try:
        yield
    finally:
        plt.switch_backend(backend)


def _algorithm(algorithm="Downwind", **kwargs):
    # Vary downwind ordering while keeping the scientific example very small.
    states = foxes.input.states.StatesTable(
        pd.DataFrame(
            {FV.WS: [7.0, 8.0, 9.0, 10.0], FV.WD: [90.0, 270.0, 90.0, 270.0]},
            index=[2, 4, 8, 16],
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for xy in ([0.0, 0.0], [600.0, 0.0]):
        farm.add_turbine(
            foxes.Turbine(xy=xy, turbine_models=["NREL5MW"]),
            verbosity=0,
        )
    return getattr(foxes.algorithms, algorithm)(
        farm,
        states,
        rotor_model="centre",
        wake_models=["Jensen_linear_k007"],
        verbosity=0,
        **kwargs,
    )


def _unexpected_grid(*args):
    # Fail in the worker if supplied point fields are ever recalculated.
    pytest.fail("grid recalculation")


def _initialize_output(output):
    # Unit-level payload tests still use the standard model initialization.
    algo = _algorithm()
    algo.initialize()
    algo.init_model(output)
    return output


class _StoreCheckingAnimation(FlowAnimation):
    def calculate(self, algo, *args, **kwargs):
        # Farm and point outputs both participate in running-state propagation.
        assert self.running and self._point_models.running
        snapshot = pickle.dumps(self)
        results = super().calculate(algo, *args, **kwargs)
        assert pickle.dumps(self) == snapshot
        return results

    def _calculate_grid(self, algo, mdata, fdata):
        # Wake-frame caches must survive until scientific output evaluation ends.
        if not algo.chunk_store:
            raise RuntimeError("Chunk store was cleared before output calculation")
        return super()._calculate_grid(algo, mdata, fdata)


class _FinalIterationAnimation(FlowAnimation):
    def initialize(self, algo, *args, **kwargs):
        # Requested outputs must not even initialize before convergence finishes.
        assert algo.final_iteration
        assert algo.iterations >= 1
        return super().initialize(algo, *args, **kwargs)

    def calculate(self, algo, *args, **kwargs):
        # Worker evaluation belongs exclusively to the final pass.
        assert algo.final_iteration and self.running
        return super().calculate(algo, *args, **kwargs)

    def update_results(self, algo, *args, **kwargs):
        # Intermediate convergence payloads must never reach requested outputs.
        assert algo.final_iteration and self.running
        return super().update_results(algo, *args, **kwargs)

    def finalize_results(self, algo, *args, **kwargs):
        # Render only the final, fully assembled result.
        assert algo.final_iteration and self.running
        return super().finalize_results(algo, *args, **kwargs)


@pytest.mark.parametrize(
    "backend",
    ["single", "numpy", "threads", "process", "multiprocess", "dask", "local_cluster"],
)
@pytest.mark.parametrize("animation_first", [False, True])
def test_real_gif_composes_without_changing_collected_results(
    tmp_path,
    backend,
    animation_first,
):
    # Optional engines remain optional outside the all-engine environment.
    module = {
        "multiprocess": "multiprocess",
        "dask": "dask",
        "local_cluster": "distributed",
    }.get(backend)
    if module is not None:
        pytest.importorskip(module)

    # Record reference results independently of diagnostic grid calculations.
    algo = _algorithm()
    with foxes.Engine.new("numpy", verbosity=0):
        reference = algo.calc_farm(outputs="farm_results_all")

    # All state chunks contribute to a single final animation per variable.
    animation = FlowAnimation(
        out_dir=tmp_path,
        figsize=(2.0, 2.0),
        grid_pars={"n_img_points": (3, 3)},
    )
    collector = FarmResultsOutput()
    outputs = [animation, collector] if animation_first else [collector, animation]
    engine_options = (
        {
            "cluster_pars": {
                "dashboard_address": None,
                "processes": False,
                "threads_per_worker": 1,
            }
        }
        if backend == "local_cluster"
        else {}
    )
    with foxes.Engine.new(
        backend,
        n_procs=2,
        chunk_size_states=2,
        chunk_size_points=4,
        verbosity=0,
        **engine_options,
    ):
        result = algo.calc_farm(outputs=outputs)
    xr.testing.assert_allclose(result[FC.FARM_RESULTS], reference)
    assert set(result) == {FC.FARM_RESULTS}
    files = list(tmp_path.glob("*.gif"))
    assert [file.name for file in files] == ["chunk_animation_WS.gif"]
    with Image.open(files[0]) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == 4
        assert gif.size == (200, 200)


@pytest.mark.parametrize(
    "chunk,states",
    [
        (None, [2, 4, 8, 16]),
        (0, [2, 4]),
        ((1, 0), [8, 16]),
        ([0, (1, 0)], [2, 4, 8, 16]),
        ([], []),
    ],
)
def test_chunk_selection_renders_once_on_parent(
    tmp_path,
    monkeypatch,
    chunk,
    states,
):
    # Rendering happens only in finalization, on the caller thread.
    parent, calls = get_ident(), []
    monkeypatch.setattr(
        FlowAnimation,
        "_render",
        lambda self, algo, payload, output_results, extra_generators=(): calls.append(
            (payload["states"], get_ident())
        ),
    )
    output = FlowAnimation(
        chunk=chunk,
        out_dir=tmp_path,
        grid_pars={"n_img_points": (2, 2)},
    )
    with foxes.Engine.new("threads", n_procs=2, chunk_size_states=2, verbosity=0):
        assert _algorithm().calc_farm(outputs=output) == {}
    assert len(calls) == bool(states)
    if calls:
        assert calls[0][0].tolist() == states
        assert calls[0][1] == parent


@pytest.mark.parametrize(
    "options,match",
    [
        ({"fps": 0}, "fps"),
        ({"fps": True}, "fps"),
        ({"fps": float("inf")}, "fps"),
        ({"fps": float("nan")}, "fps"),
        ({"layout_pars": {"ax": None}}, "layout_pars"),
        ({"base_name": "file.png"}, "base_name"),
        ({"chunk": -1}, "chunk"),
        ({"chunk": (0, 1, 2)}, "chunk"),
        ({"chunk": True}, "chunk"),
        ({"states_sel": [0], "states_isel": [0]}, "states_sel"),
        ({"states_isel": [-1]}, "positions"),
        ({"variables": []}, "variables"),
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, options, match):
    # Construction is cheap; initialization rejects settings before loading data.
    output = FlowAnimation(out_dir=tmp_path, **options)
    assert not output.initialized
    with pytest.raises(ValueError, match=match):
        output.initialize(None)


@pytest.mark.parametrize(
    "output_verbosity,algo_verbosity,announces",
    [
        (None, 0, False),
        (None, 1, True),
        (None, 2, True),
        (None, 3, True),
        (0, 1, False),
        (0, 3, False),
        (1, 0, True),
        (1, 3, True),
        (2, 0, True),
        (3, 0, True),
    ],
)
@pytest.mark.parametrize(
    "animation_class", [FlowAnimation, foxes.models.outputs.FlowTurbineAnimation]
)
def test_writing_message_verbosity(
    tmp_path, capsys, output_verbosity, algo_verbosity, announces, animation_class
):
    # Output verbosity overrides algorithm verbosity for either renderer.
    algo = _algorithm()
    algo.verbosity = algo_verbosity
    animation = animation_class(
        out_dir=tmp_path,
        verbosity=output_verbosity,
        fps=10,
        grid_pars={"n_img_points": (2, 2)},
        figsize=(2, 2),
    )
    with foxes.Engine.new("single", verbosity=0):
        algo.calc_farm(outputs=animation)
    text = capsys.readouterr().out
    path = tmp_path / "chunk_animation_WS.gif"
    level = algo_verbosity if output_verbosity is None else output_verbosity
    assert (f"Creating animation {path}" in text) is announces
    assert (f"Writing {path}" in text) is (level > 1)
    assert (f"Finished writing {path}" in text) is (level > 1)
    assert ("Created 4 animation frames." in text) is (level > 1)
    assert ("Encoding 4 frames at 10 fps with pillow" in text) is (level > 1)
    frames = [line.strip() for line in text.splitlines() if line.startswith("  Frame ")]
    assert frames == ([f"Frame {i}" for i in range(4)] if level > 2 else [])
    assert text.count(": Encoding frame ") == (4 if level > 2 else 0)
    if level > 1:
        assert text.index("Creating animation") < text.index("Writing ")
        assert text.index("Writing ") < text.index("Finished writing")
    elif not announces and algo_verbosity == 0:
        assert not text


@pytest.mark.parametrize(
    "selection,expected",
    [
        ({"states_sel": [4, 16]}, [4, 16]),
        ({"states_isel": [1, 3]}, [4, 16]),
        ({"states_sel": [16]}, [16]),
        ({"states_isel": []}, []),
    ],
)
@pytest.mark.parametrize("backend", ["numpy", "threads", "process"])
def test_state_selection_is_global(tmp_path, monkeypatch, selection, expected, backend):
    # State positions refer to the input states, not each worker chunk.
    calls = []
    monkeypatch.setattr(
        FlowAnimation,
        "_render",
        lambda self, algo, payload, output_results, extra_generators=(): calls.append(
            payload["states"]
        ),
    )
    output = FlowAnimation(
        out_dir=tmp_path,
        grid_pars={"n_img_points": (2, 2)},
        **selection,
    )
    with foxes.Engine.new(backend, n_procs=2, chunk_size_states=2, verbosity=0):
        _algorithm().calc_farm(outputs=output)
    assert [s.tolist() for s in calls] == ([expected] if expected else [])


def _target_chunk(output, state_chunk, target_chunk):
    # Use multiple target points and broadcast fields to expose shape mistakes.
    states = np.array([4 * state_chunk, 4 * state_chunk + 2])
    mdata = MData(
        {FC.STATE: states},
        {FC.STATE: (FC.STATE,)},
        chunki_states=state_chunk,
        chunki_points=target_chunk,
    )
    mdata.add(output._order_key, states.copy(), (FC.STATE,))
    points = np.zeros((2, 2, 2, 3))
    points[..., 0] = (target_chunk * 4 + np.arange(4)).reshape(1, 2, 2)
    tdata = TData.from_tpoints(points, np.array([0.5, 0.5]), mdata=mdata)
    tdata.add(
        FV.WS, np.full((2, 2, 2), 8.0 + state_chunk), (FC.STATE, FC.TARGET, FC.TPOINT)
    )
    tdata.add(FV.WD, np.array([[[270.0]]]), (FC.STATE, FC.TARGET, FC.TPOINT))
    return mdata, FData(), tdata


def test_target_tiles_are_copied_buffered_and_combined(tmp_path, monkeypatch):
    # Existing targets never trigger a second scientific calculation.
    output = _initialize_output(FlowAnimation(out_dir=tmp_path))
    monkeypatch.setattr(FlowAnimation, "_calculate_grid", _unexpected_grid)
    calls, results = [], output.get_initial_output_results(None)
    monkeypatch.setattr(
        output,
        "_render",
        lambda algo, payload, output_results, extra_generators=(): calls.append(
            payload
        ),
    )
    for key in [(1, 1), (0, 1), (1, 0), (0, 0)]:
        md, fd, td = _target_chunk(output, *key)
        before = deepcopy(td)
        chunk = output.calculate(None, md, fd, td, {})
        for variable in td:
            np.testing.assert_array_equal(td[variable], before[variable])
        assert not np.shares_memory(chunk[output._result_key]["points"], td[FC.TARGETS])
        pending = Dict(
            {output._result_key: {key: chunk[output._result_key]}, "other": 1}
        )
        output.update_results(None, 2, 2, pending, results)
        assert pending == {"other": 1}
        assert not calls
    results["collector"] = "untouched"
    output.finalize_results(None, 2, 2, results)
    assert results == {"collector": "untouched"}
    assert len(calls) == 1
    payload = calls[0]
    assert payload["states"].tolist() == [0, 2, 4, 6]
    assert payload["points"].shape == (4, 8, 3)
    np.testing.assert_array_equal(payload["points"][0, :, 0], np.arange(8))
    np.testing.assert_array_equal(payload["values"][FV.WS][:, 0], [8, 8, 9, 9])


def test_grid_is_loaded_once_and_point_blocks_are_bounded(tmp_path, monkeypatch):
    # Inspect the real scientific chain without storing runtime state on models.
    output = FlowAnimation(
        out_dir=tmp_path,
        grid_pars={"n_img_points": (3, 3)},
    )
    algo, calls, rendered = _algorithm(), [], []
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=4, verbosity=0
    ):
        algo.initialize()
        algo.init_model(output)
        chain = output._point_models
        calculate = type(chain).calculate

        def capture(self, algo, md, fd, td, *args, **kwargs):
            if self is chain:
                calls.append((td.n_targets, md.extra_data[output._grid_key]))
                assert self.running and output.running
                assert td[FC.TARGETS].strides[0] == 0
            return calculate(self, algo, md, fd, td, *args, **kwargs)

        monkeypatch.setattr(type(chain), "calculate", capture)
        monkeypatch.setattr(
            output,
            "_render",
            lambda algo, payload, output_results, extra_generators=(): rendered.append(
                payload
            ),
        )
        algo.calc_farm(outputs=output)
    assert [n for n, _ in calls] == [4, 4, 1, 4, 4, 1]
    assert all(points is calls[0][1] for _, points in calls)
    assert calls[0][1].shape == (9, 3)
    assert rendered[0]["points"].shape == (4, 9, 3)


@pytest.mark.parametrize("ambient", [False, True])
def test_generated_grid_matches_independent_point_calculation(
    tmp_path,
    monkeypatch,
    ambient,
):
    # Check wake values and alternating turbine order, not just file existence.
    algo, payloads = _algorithm(), []
    output = FlowAnimation(
        out_dir=tmp_path,
        grid_pars={"n_img_points": (3, 3)},
        ambient=ambient,
    )
    monkeypatch.setattr(
        output,
        "_render",
        lambda algo, payload, output_results, extra_generators=(): payloads.append(
            payload
        ),
    )
    with foxes.Engine.new(
        "numpy",
        chunk_size_states=2,
        chunk_size_points=4,
        verbosity=0,
    ):
        farm = algo.calc_farm(
            outputs=[FarmResultsOutput(), output],
            ambient=ambient,
        )[FC.FARM_RESULTS]
        payload = payloads[0]
        points = algo.calc_points(
            farm,
            payload["points"][0],
            ambient=ambient,
        )
    for variable in (FV.WS, FV.WD):
        np.testing.assert_allclose(payload["values"][variable], points[variable])


def test_calc_outputs_uses_output_calculation_points(tmp_path, monkeypatch):
    # calc_outputs obtains the grid from the output and delegates point chunking.
    algo = _algorithm()
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        farm = algo.calc_farm(outputs="farm_results_all")

    output = FlowAnimation(out_dir=tmp_path, grid_pars={"n_img_points": (2, 2)})
    payloads = []
    monkeypatch.setattr(
        output,
        "_render",
        lambda algo, payload, output_results, extra_generators=(): payloads.append(
            payload
        ),
    )
    outputs = foxes.core.OutputList(outputs=[output])
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        algo.calc_outputs(farm_results=farm, outputs=outputs)

    assert len(payloads) == 1
    assert payloads[0]["points"].shape == (4, 4, 3)
    assert payloads[0]["states"].tolist() == [2, 4, 8, 16]


def test_output_list_rejects_different_calculation_points(tmp_path):
    # One calc_points run cannot satisfy outputs with different point grids.
    outputs = foxes.core.OutputList(
        outputs=[
            FlowAnimation(
                out_dir=tmp_path / "first", grid_pars={"n_img_points": (2, 2)}
            ),
            FlowAnimation(
                out_dir=tmp_path / "second", grid_pars={"n_img_points": (3, 3)}
            ),
        ]
    )
    algo = _algorithm()
    algo.initialize()
    algo.init_model(outputs)

    with pytest.raises(ValueError, match="different calculation points"):
        outputs.get_calc_points(algo)


@pytest.mark.parametrize("backend", ["numpy", "threads", "process"])
def test_calc_points_uses_existing_fields(tmp_path, monkeypatch, backend):
    # Public point calculations retain their dataset while writing one animation.
    algo = _algorithm()
    with foxes.Engine.new("numpy", verbosity=0):
        farm = algo.calc_farm(outputs="farm_results_all")
    points = np.array([[0.0, 0.0, 90.0], [400.0, 50.0, 90.0], [800.0, 0.0, 90.0]])
    with foxes.Engine.new("numpy", verbosity=0):
        expected = algo.calc_points(farm, points)
    output = _StoreCheckingAnimation(out_dir=tmp_path, figsize=(2.0, 2.0))
    monkeypatch.setattr(FlowAnimation, "_calculate_grid", _unexpected_grid)
    with foxes.Engine.new(
        backend,
        n_procs=2,
        chunk_size_states=2,
        chunk_size_points=2,
        verbosity=0,
    ):
        actual = algo.calc_points(
            farm,
            points,
            outputs=[foxes.models.outputs.PointResultsOutput(), output],
        )[FC.POINT_RESULTS]
    xr.testing.assert_allclose(actual, expected)
    with Image.open(tmp_path / "chunk_animation_WS.gif") as gif:
        assert gif.n_frames == 4


@pytest.mark.parametrize(
    "selection,match",
    [
        ({"chunk": 9}, "Selected chunk"),
        ({"states_sel": [999]}, "state labels"),
        ({"states_isel": [4]}, "State position"),
    ],
)
def test_unavailable_selections_fail(tmp_path, selection, match):
    # Invalid selections are diagnosed instead of producing no animation.
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        with pytest.raises(ValueError, match=match):
            _algorithm().calc_farm(
                outputs=FlowAnimation(
                    out_dir=tmp_path,
                    grid_pars={"n_img_points": (2, 2)},
                    **selection,
                )
            )


def test_multiple_outputs_have_private_buffers(tmp_path):
    # Independently configured animations must not consume each other's buffers.
    first, second = [
        _initialize_output(FlowAnimation(out_dir=tmp_path)) for _ in range(2)
    ]
    assert first._result_key != second._result_key
    pending = Dict({first._result_key: {(0, 0): {}}, second._result_key: {(0, 0): {}}})
    results = Dict()
    first.update_results(None, 1, 1, pending, results)
    assert set(pending) == {second._result_key}
    assert set(results) == {first._result_key}


@pytest.mark.parametrize("point_run", [False, True])
@pytest.mark.parametrize("verbosity", [0, 1, 2, 3])
def test_save_failure_propagates_and_closes_figure(
    tmp_path, monkeypatch, capsys, point_run, verbosity
):
    # Finalization cannot report success or leak figures after a writer failure.
    import matplotlib.pyplot as plt
    from matplotlib.animation import Animation

    before = plt.get_fignums()

    def fail(*args, **kwargs):
        raise OSError("synthetic animation write failure")

    monkeypatch.setattr(Animation, "save", fail)
    algo = _algorithm()
    output = FlowAnimation(
        out_dir=tmp_path,
        verbosity=verbosity,
        grid_pars={"n_img_points": (2, 2)},
    )
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        if point_run:
            farm = algo.calc_farm(outputs="farm_results_all")
        with pytest.raises(OSError, match="synthetic animation"):
            if point_run:
                algo.calc_points(farm, np.array([[400.0, 0.0, 90.0]]), outputs=output)
            else:
                algo.calc_farm(outputs=output)
    assert plt.get_fignums() == before
    assert not algo.running and not output.running
    text = capsys.readouterr().out
    assert ("Writing " in text) is (verbosity > 1)
    assert "Finished writing" not in text


def test_missing_target_fields_are_contextual(tmp_path):
    # Missing external fields must identify the offending animation variable.
    output = _initialize_output(FlowAnimation(out_dir=tmp_path, variables=[FV.TI]))
    md, fd, td = _target_chunk(output, 0, 0)
    with pytest.raises(KeyError, match="Missing target field 'TI'"):
        output.calculate(None, md, fd, td, {})


@pytest.mark.parametrize("backend", ["numpy", "process", "dask"])
def test_scientific_output_keeps_chunk_store_available(tmp_path, backend):
    # Generated flow fields may reuse wake-frame data from the farm calculation.
    if backend == "dask":
        pytest.importorskip("dask")
    output = _StoreCheckingAnimation(
        out_dir=tmp_path,
        figsize=(2.0, 2.0),
        grid_pars={"n_img_points": (2, 2)},
    )
    with foxes.Engine.new(
        backend,
        n_procs=2,
        chunk_size_states=2,
        chunk_size_points=2,
        verbosity=0,
    ):
        _algorithm().calc_farm(outputs=output)


def test_streamline_cache_reuse_matches_collected_farm(tmp_path, monkeypatch):
    # Cached streamline geometry is required during generated-grid evaluation.
    algo = _algorithm(
        algorithm="Iterative",
        wake_frame="streamlines_100",
        max_wake_length_km=1.0,
    )
    payloads = []
    output = _FinalIterationAnimation(
        out_dir=tmp_path,
        grid_pars={"n_img_points": (3, 3)},
    )
    monkeypatch.setattr(
        output,
        "_render",
        lambda algo, payload, output_results, extra_generators=(): payloads.append(
            payload
        ),
    )
    with foxes.Engine.new(
        "numpy",
        chunk_size_states=2,
        chunk_size_points=4,
        verbosity=0,
    ):
        farm = algo.calc_farm(outputs=[FarmResultsOutput(), output])[FC.FARM_RESULTS]
        payload = payloads[0]
        expected = algo.calc_points(farm, payload["points"][0])
        independent = _algorithm(
            wake_frame="streamlines_100", max_wake_length_km=1.0
        ).calc_points(farm, payload["points"][0])
    np.testing.assert_allclose(payload["values"][FV.WS], expected[FV.WS])
    np.testing.assert_allclose(payload["values"][FV.WS], independent[FV.WS])


@pytest.mark.parametrize("backend", ["numpy", "threads", "process"])
def test_iterative_writes_only_after_convergence(tmp_path, backend):
    # Real encodings must contain final states once, not repeated iterations.
    output = _FinalIterationAnimation(
        out_dir=tmp_path,
        figsize=(2.0, 2.0),
        grid_pars={"n_img_points": (2, 2)},
    )
    with foxes.Engine.new(
        backend,
        n_procs=2,
        chunk_size_states=2,
        chunk_size_points=2,
        verbosity=0,
    ):
        results = _algorithm(algorithm="Iterative").calc_farm(
            outputs=[FarmResultsOutput(), output],
        )
    assert results[FC.FARM_RESULTS].sizes[FC.STATE] == 4
    with Image.open(tmp_path / "chunk_animation_WS.gif") as gif:
        assert gif.n_frames == 4
    assert not output.running and not output._point_models.running


def test_missing_tiles_are_rejected(tmp_path):
    # Finalization cannot silently create an incomplete animation.
    output = _initialize_output(FlowAnimation(out_dir=tmp_path))
    md, fd, td = _target_chunk(output, 0, 0)
    chunk = output.calculate(None, md, fd, td, {})[output._result_key]
    results = output.get_initial_output_results(None)
    results[output._result_key] = {(0, 0): chunk}
    with pytest.raises(ValueError, match="Missing slice chunks"):
        output.finalize_results(None, 1, 2, results)


def test_moving_target_artists_preserve_coordinates_and_units(tmp_path):
    # Moving targets retain their true coordinates and a comparable colour scale.
    import matplotlib.pyplot as plt

    output = FlowAnimation(out_dir=tmp_path)
    payload = dict(
        states=np.array([5, 2]),
        points=np.array([[[0.0, 0.0, 90.0]], [[10.0, 20.0, 90.0]]]),
        values={FV.WS: np.array([[8.0], [9.0]])},
    )
    fig, ax = plt.subplots()
    try:
        frames = list(
            output._scatter_frames(
                fig,
                ax,
                payload,
                FV.WS,
                {"vmin": None, "vmax": None},
                {"vlabel": "WS [m/s]"},
            )
        )
        assert len(frames) == 2
        for index, (_, artists) in enumerate(frames):
            np.testing.assert_array_equal(
                artists[0].get_offsets(),
                payload["points"][index, :, :2],
            )
            assert artists[0].get_clim() == (8.0, 9.0)
        assert ax.get_xlabel() == "x [m]"
        assert ax.get_ylabel() == "y [m]"
        assert fig.axes[1].get_ylabel() == "WS [m/s]"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("point_run", [False, True])
def test_default_engine_supports_animation(tmp_path, point_run):
    # Algorithm-owned default engines are opened after running-state propagation.
    algo = _algorithm()
    output = FlowAnimation(
        out_dir=tmp_path,
        figsize=(2.0, 2.0),
        grid_pars={"n_img_points": (2, 2)},
    )
    if point_run:
        with foxes.Engine.new("numpy", verbosity=0):
            farm = algo.calc_farm(outputs="farm_results_all")
        algo.calc_points(farm, np.array([[400.0, 0.0, 90.0]]), outputs=output)
    else:
        algo.calc_farm(outputs=output)
    with Image.open(tmp_path / "chunk_animation_WS.gif") as gif:
        assert gif.n_frames == 4
    assert not output.running


def test_invalid_point_limit_unwinds_started_models(tmp_path):
    # A rejected running hook must not strand the algorithm in running state.
    algo = _algorithm()
    output = FlowAnimation(out_dir=tmp_path)
    with foxes.Engine.new("numpy", chunk_size_points=0, verbosity=0):
        with pytest.raises(ValueError, match="chunk_size_points must be positive"):
            algo.calc_farm(outputs=output)
    assert not algo.running and not output.running


@pytest.mark.parametrize(
    "name,chunk,base_name",
    [
        ("create_gif", None, "ani.gif"),
        ("create_mp4", None, "ani.mp4"),
        ("create_turbine_gif", None, "ani.gif"),
        ("create_turbine_mp4", None, "ani.mp4"),
        ("chunk_gif_1", 1, "ani.gif"),
        ("chunk_mp4_1", 1, "ani.mp4"),
    ],
)
def test_model_book_defers_animation_preparation(monkeypatch, name, chunk, base_name):
    # Registering defaults must not build grids or initialize scientific models.
    monkeypatch.setattr(
        "foxes.models.outputs.grids.get_grid_xy",
        lambda *args, **kwargs: pytest.fail("grid generated during construction"),
    )
    output = foxes.ModelBook().outputs[name]
    assert isinstance(output, FlowAnimation)
    assert output.chunk == chunk
    assert output.base_name == base_name
    assert not output.initialized
    assert output.sub_models() == []
    assert not hasattr(output, "_result_key")
    assert not hasattr(output, "_variables")
    if "turbine" in name:
        assert isinstance(output, foxes.models.outputs.FlowTurbineAnimation)
        assert output.turbines is None


@pytest.mark.parametrize(
    "name,frames",
    [
        ("create_gif", 4),
        ("create_mp4", 4),
        ("create_turbine_gif", 4),
        ("create_turbine_mp4", 4),
        ("chunk_gif_1", 2),
        ("chunk_mp4_1", 2),
    ],
)
def test_registered_animation_can_be_selected_by_name(tmp_path, name, frames):
    # Optional movie encoders do not prevent testing the GIF registry paths.
    if "mp4" in name and which("ffmpeg") is None:
        pytest.skip("MP4 encoding requires ffmpeg")

    # Configure the registered instance before initialization, then resolve its name.
    algo = _algorithm()
    output = algo.mbook.outputs[name]
    output.out_dir = tmp_path
    output.figsize = (2.0, 2.0)
    output.grid_pars = {"n_img_points": (2, 2)}
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        assert algo.calc_farm(outputs=name) == {}
    if "gif" in name:
        with Image.open(tmp_path / "ani_WS.gif") as gif:
            assert gif.n_frames == frames
    else:
        assert (tmp_path / "ani_WS.mp4").read_bytes()[4:8] == b"ftyp"


def test_initialize_snapshots_mutable_configuration(tmp_path):
    # External settings can change before initialization but cannot alter a run.
    variables = [FV.WS]
    grid_pars = {"n_img_points": (2, 2)}
    algo = _algorithm()
    algo.initialize()
    output = FlowAnimation(
        out_dir=tmp_path,
        variables=variables,
        grid_pars=grid_pars,
    )
    algo.init_model(output)
    variables.append(FV.TI)
    grid_pars["n_img_points"] = (8, 8)
    assert output._variables == (FV.WS,)
    assert algo.loaded_data["extra_data"][output._grid_key].shape == (4, 3)


@pytest.mark.parametrize("keep_stash", [False, True])
def test_running_model_does_not_retain_grid_or_large_selection(tmp_path, keep_stash):
    # State-sized selections and grids belong in loaded data, not worker models.
    count = 20000
    states = foxes.input.states.StatesTable(
        pd.DataFrame({FV.WS: np.full(count, 8.0), FV.WD: np.full(count, 270.0)}),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
    )
    algo = foxes.algorithms.Downwind(
        _algorithm().farm,
        states,
        wake_models=[],
        verbosity=0,
    )
    selection = list(range(0, count, 2))
    output = FlowAnimation(
        out_dir=tmp_path,
        states_isel=selection,
        grid_pars={"n_img_points": (200, 200)},
    )
    algo.initialize()
    algo.init_model(output)
    extra = algo.loaded_data["extra_data"]
    assert extra[output._grid_key].nbytes == 960000
    dims, mask = algo.loaded_data["data_vars"][output._selection_key]
    assert dims == (FC.STATE,)
    assert mask.nbytes == count
    assert mask.sum() == len(selection)
    stash = {} if keep_stash else None
    with foxes.Engine.new("numpy", verbosity=0):
        output.set_running(algo, stash)
        try:
            assert len(pickle.dumps(output)) < 32768
            assert not hasattr(output, "states_isel")
            assert not hasattr(output, "_states_isel")
            assert not hasattr(output, "_grid_pars")
        finally:
            output.unset_running(algo, stash)
    if keep_stash:
        assert output.states_isel is selection


@pytest.mark.parametrize("output_type", [FlowAnimation, FlowTurbineAnimation])
@pytest.mark.parametrize("failure", [False, True])
def test_animation_owns_flat_render_inputs(tmp_path, monkeypatch, output_type, failure):
    # Unpicklable options must remain available only in the parent's result mapping.
    output = output_type(
        out_dir=tmp_path,
        grid_pars={"n_img_points": (2, 2)},
        plot_pars={"renderer_only": lambda: None},
        layout_pars={"annotate": 1},
    )
    plot_pars, layout_pars = output.plot_pars, output.layout_pars
    write_pars, normalize = output.write_pars, output.normalize
    results_seen = []

    def render(self, algo, payload, output_results, extra_generators=()):
        assert output.running
        assert not hasattr(output, "plot_pars")
        assert not hasattr(output, "layout_pars")
        assert not hasattr(output, "write_pars")
        assert not hasattr(output, "normalize")
        assert output_results[output.var("plot_pars")] is plot_pars
        assert output_results[output.var("layout_pars")] is layout_pars
        assert output_results[output.var("write_pars")] is write_pars
        assert output_results[output.var("normalize")] is normalize
        if isinstance(output, FlowTurbineAnimation):
            assert not hasattr(output, "turbines")
            assert not hasattr(output, "_turbines")
            assert not hasattr(output, "ref_xy")
            assert not hasattr(output, "_ref_xy")
            assert output_results[output.var("_turbines")] == (0, 1)
            assert output_results[output.var("_ref_xy")] is None
        assert len(pickle.dumps(output)) < 32768
        results_seen.append(output_results)
        if failure:
            raise RuntimeError("synthetic renderer failure")

    monkeypatch.setattr(output_type, "_render", render)
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        if failure:
            with pytest.raises(RuntimeError, match="synthetic renderer failure"):
                _algorithm().calc_farm(outputs=output)
        else:
            assert _algorithm().calc_farm(outputs=output) == {}
            assert results_seen == [{}]
    assert len(results_seen) == 1
    assert not output.running
    assert output.plot_pars is plot_pars
    assert output.layout_pars is layout_pars
    assert output.write_pars is write_pars
    assert output.normalize is normalize
    if isinstance(output, FlowTurbineAnimation):
        assert output.turbines is None
        assert output._turbines == (0, 1)


@pytest.mark.parametrize(
    "options,mask",
    [
        ({}, None),
        ({"states_sel": [2, 4, 8, 16]}, None),
        ({"states_isel": [0, 1, 2, 3]}, None),
        ({"states_sel": [4, 16]}, [False, True, False, True]),
        ({"states_isel": []}, [False, False, False, False]),
    ],
)
def test_state_selection_is_loaded_only_when_partial(tmp_path, options, mask):
    # Full selection uses None; partial masks have exactly the state dimension.
    algo = _algorithm()
    algo.initialize()
    output = FlowAnimation(out_dir=tmp_path, **options)
    algo.init_model(output)
    loaded = algo.loaded_data
    if mask is None:
        assert output._selection is None
        assert output._selection_key not in loaded["data_vars"]
    else:
        assert output._selection == output._selection_key
        dims, actual = loaded["data_vars"][output._selection]
        assert dims == (FC.STATE,)
        assert actual.dtype == np.dtype(bool)
        np.testing.assert_array_equal(actual, mask)


def test_datetime_state_selection_uses_input_labels(tmp_path):
    # Calendar labels must compare as timestamps, not integer nanosecond values.
    algo = _algorithm()
    labels = pd.date_range("2020-01-01", periods=4)
    algo.states.data_source.index = labels
    algo.initialize()
    output = FlowAnimation(out_dir=tmp_path, states_sel=[labels[1]])
    algo.init_model(output)
    _, selection = algo.loaded_data["data_vars"][output._selection]
    np.testing.assert_array_equal(selection, [False, True, False, False])


def test_force_initialization_removes_obsolete_selection_mask(tmp_path):
    # Replacing a partial selection with all states must leave no stale variable.
    algo = _algorithm()
    algo.initialize()
    output = FlowAnimation(out_dir=tmp_path, states_isel=[1])
    algo.init_model(output)
    assert output._selection_key in algo.loaded_data["data_vars"]
    output.states_isel = [0, 1, 2, 3]
    output.initialize(algo, loaded_data=algo.loaded_data, force=True)
    assert output._selection is None
    assert output._selection_key not in algo.loaded_data["data_vars"]


@pytest.mark.parametrize("variables", [[FV.WS], [FV.WS, FV.WD]])
@pytest.mark.parametrize(
    "animation_class", [FlowAnimation, foxes.models.outputs.FlowTurbineAnimation]
)
def test_filename_transform_and_multiple_variables(
    tmp_path, variables, animation_class
):
    # Each final variable file passes through the standard Output path helper.
    output = animation_class(
        out_dir=tmp_path,
        variables=variables,
        figsize=(2.0, 2.0),
        grid_pars={"n_img_points": (2, 2)},
        out_fname_fun=lambda path: path.with_name("renamed_" + path.name),
    )
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        _algorithm().calc_farm(outputs=output)
    expected = sorted(f"renamed_chunk_animation_{v}.gif" for v in variables)
    assert sorted(p.name for p in tmp_path.glob("*.gif")) == expected


@pytest.mark.skipif(which("ffmpeg") is None, reason="MP4 encoding requires ffmpeg")
@pytest.mark.parametrize("transform_suffix", [False, True])
@pytest.mark.parametrize("verbosity", [0, 1, 2, 3])
def test_real_mp4_encoding(tmp_path, capsys, transform_suffix, verbosity):
    # Exercise the optional encoder, not just its selected writer name.
    output = FlowAnimation(
        out_dir=tmp_path,
        base_name="flow.gif" if transform_suffix else "flow.mp4",
        verbosity=verbosity,
        fps=10,
        figsize=(2.0, 2.0),
        grid_pars={"n_img_points": (2, 2)},
        out_fname_fun=(
            (lambda path: path.with_suffix(".mp4")) if transform_suffix else None
        ),
    )
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        _algorithm().calc_farm(outputs=output)
    encoded = (tmp_path / "flow_WS.mp4").read_bytes()
    assert encoded[4:8] == b"ftyp"
    assert len(encoded) > 1024
    text = capsys.readouterr().out
    assert (f"Creating animation {tmp_path / 'flow_WS.mp4'}" in text) is (verbosity > 0)
    assert (f"Writing {tmp_path / 'flow_WS.mp4'}" in text) is (verbosity > 1)
    assert (f"Finished writing {tmp_path / 'flow_WS.mp4'}" in text) is (verbosity > 1)
    assert ("Encoding 4 frames at 10 fps with ffmpeg" in text) is (verbosity > 1)
    assert text.count(": Encoding frame ") == (4 if verbosity > 2 else 0)
