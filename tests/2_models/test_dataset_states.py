from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.algorithms.downwind.models.prepare_for_output import PrepareForOutput
from foxes.models.outputs import FarmResultsOutput, SliceData
from foxes.core import FData, MData
from foxes.input.states import DatasetStates, FieldData


def _field_dataset():
    time = np.arange(10, 14)
    height = np.arange(0.0, 401.0, 50.0)
    y = np.arange(0.0, 6000.0, 1000.0)
    x = np.arange(0.0, 8000.0, 1000.0)
    wind = (
        8.0
        + 0.25 * time[:, None, None, None]
        + 0.001 * height[None, :, None, None]
        + 0.0001 * y[None, None, :, None]
        + 0.0002 * x[None, None, None, :]
    )
    return xr.Dataset(
        {"ws": (("time", "height", "y", "x"), wind)},
        coords={"time": time, "height": height, "y": y, "x": x},
    )


def _field_algorithm(source, xy, **selection):
    states = FieldData(
        source,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        states_coord="time",
        time_format=None,
        x_coord="x",
        y_coord="y",
        var2ncvar={FV.WS: "ws"},
        fixed_vars={FV.WD: 270.0, FV.TI: 0.08, FV.RHO: 1.225},
        bounds_extra_space=0.0,
        height_bounds=selection.pop("height_bounds", (80.0, 120.0)),
        **selection,
    )
    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(xy=xy, H=100.0, turbine_models=["NREL5MW"]), verbosity=0
    )
    return foxes.algorithms.Downwind(farm, states, wake_models=[], verbosity=0)


@pytest.mark.parametrize("engine_type", ["numpy", "process"])
def test_dataset_field_crops_prepared_arrays_and_worker_chunks(
    engine_type, tmp_path, monkeypatch
):
    source = _field_dataset()
    original = source.copy(deep=True)
    source["ws"].values.setflags(write=False)
    path = tmp_path / "field.nc"
    source.to_netcdf(path)
    with foxes.Engine.new(
        engine_type, n_procs=2, chunk_size_states=2, verbosity=0
    ) as engine:
        for xy, x0, y0 in (
            ([1250.0, 1250.0], 1000, 1000),
            ([3250.0, 2250.0], 3000, 2000),
        ):
            algo = _field_algorithm(source, xy)
            algo.initialize()
            states = algo.states
            data_key = states.var("data0")
            model_data, _ = algo.get_model_data()
            expected = source.sel(
                x=slice(x0, x0 + 1000),
                y=slice(y0, y0 + 1000),
                height=slice(50, 150),
            )["ws"].transpose("time", "x", "y", "height")
            np.testing.assert_array_equal(
                model_data[data_key].values[..., 0], expected.values
            )
            assert model_data[data_key].nbytes == expected.nbytes < source["ws"].nbytes
            chunks = []
            original_chunk = engine.get_chunk_input_data

            def capture_chunk(*args, **kwargs):
                chunk = original_chunk(*args, **kwargs)
                chunks.append(chunk[0][data_key].shape)
                return chunk

            with monkeypatch.context() as patch:
                patch.setattr(engine, "get_chunk_input_data", capture_chunk)
                calculated = algo.calc_farm()
            assert chunks == [(2, 2, 2, 3, 1)] * 2
            reference = _field_algorithm(path, xy, load_mode="fly").calc_farm()
            xr.testing.assert_allclose(calculated, reference)
    xr.testing.assert_identical(source, original)
    assert not source["ws"].values.flags.writeable


def test_dataset_field_applies_explicit_selections_once():
    source = _field_dataset()
    isel = {"time": slice(1, 4, 2), "x": slice(1, 7, 2)}
    original_isel = deepcopy(isel)
    sel = {"x": slice(2000.0, 6000.0)}
    algo = _field_algorithm(source, [3250.0, 2250.0], isel=isel, sel=sel)
    with foxes.Engine.new("numpy", n_procs=1, verbosity=0):
        algo.initialize()
        model_data, _ = algo.get_model_data()
        expected = (
            source.isel(original_isel)
            .sel(**sel, y=slice(2000, 3000), height=slice(50, 150))["ws"]
            .transpose("time", "x", "y", "height")
        )
        actual = model_data[algo.states.var("data0")]
        np.testing.assert_array_equal(actual.values[..., 0], expected.values)
        np.testing.assert_array_equal(model_data[FC.STATE], [11, 13])
        assert actual.shape == (2, 2, 2, 3, 1)
        result = algo.calc_farm()
    assert result.sizes[FC.STATE] == 2


@pytest.mark.parametrize("engine_type", ["numpy", "process"])
@pytest.mark.parametrize("source_kind", ["dataset", "preload", "fly"])
def test_dataset_field_disables_automatic_farm_cropping(
    engine_type, source_kind, tmp_path
):
    source = _field_dataset()
    original = source.copy(deep=True)
    if source_kind == "dataset":
        data_source = source
        load_mode = "preload"
    else:
        data_source = tmp_path / "field.nc"
        source.to_netcdf(data_source)
        load_mode = source_kind
    algo = _field_algorithm(
        data_source,
        [1250.0, 1250.0],
        crop_to_farm=False,
        height_bounds=None,
        load_mode=load_mode,
    )
    output = SliceData(
        variables=[FV.WS],
        grid_pars=dict(
            xmin=6000.0,
            xmax=7000.0,
            ymin=4000.0,
            ymax=5000.0,
            z=350.0,
            n_img_points=(2, 2),
        ),
    )
    with foxes.Engine.new(
        engine_type, n_procs=2, chunk_size_states=2, chunk_size_points=3, verbosity=0
    ):
        farm = algo.calc_farm()
        result = algo.calc_outputs(output, farm_results=farm)
        points = algo.calc_points(farm, np.array([[7000.0, 5000.0, 400.0]]))
    expected = source.ws.sel(x=[6000.0, 7000.0], y=[4000.0, 5000.0], height=350.0)
    np.testing.assert_allclose(result[FV.WS], expected.values)
    np.testing.assert_allclose(
        points[FV.WS][:, 0], source.ws.sel(x=7000.0, y=5000.0, height=400.0)
    )
    xr.testing.assert_identical(source, original)


def test_disabled_farm_cropping_preserves_explicit_spatial_selections():
    source = _field_dataset()
    algo = _field_algorithm(
        source,
        [3250.0, 2250.0],
        crop_to_farm=False,
        isel={"time": [1, 3], "x": slice(3, 7)},
        sel={"y": slice(2000.0, 4000.0)},
    )
    with foxes.Engine.new("numpy", verbosity=0):
        algo.initialize()
        model_data, _ = algo.get_model_data()
        expected = (
            source.isel(time=[1, 3], x=slice(3, 7))
            .sel(y=slice(2000.0, 4000.0), height=slice(50.0, 150.0))
            .ws.transpose("time", "x", "y", "height")
        )
        np.testing.assert_array_equal(
            model_data[algo.states.var("data0")].values[..., 0], expected.values
        )
        algo.finalize()


@pytest.mark.parametrize("crop_to_farm", ["false", 0, None])
def test_dataset_states_rejects_nonboolean_cropping(crop_to_farm):
    with pytest.raises(ValueError, match="crop_to_farm must be a boolean"):
        DatasetStates(xr.Dataset(), output_vars=[], crop_to_farm=crop_to_farm)


@pytest.mark.parametrize("reorder", [False, True])
def test_reorder_farm_output_derives_inverse_order(reorder):
    order = np.array([[2, 0, 1], [1, 2, 0]])
    fdata = FData(
        data={
            FV.ORDER: order,
            FV.WS: np.array([[30.0, 10.0, 20.0], [50.0, 60.0, 40.0]]),
        },
        dims={
            FV.ORDER: (FC.STATE, FC.TURBINE),
            FV.WS: (FC.STATE, FC.TURBINE),
        },
    )
    model = PrepareForOutput(
        output=FarmResultsOutput(variables=[FV.WS]), reorder=reorder
    )

    model.calculate(algo=None, mdata=MData(), fdata=fdata)

    np.testing.assert_allclose(
        fdata[FV.WS],
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
        if reorder
        else [[30.0, 10.0, 20.0], [50.0, 60.0, 40.0]],
    )


def test_turbine_only_data_follows_each_states_downwind_order():
    states = DatasetStates(
        data_source=xr.Dataset(),
        output_vars=[FV.WS],
        load_mode="preload",
    )
    states._cmap = {FC.STATE: FC.STATE}
    states.META = states.var("meta")
    data_key = states.var("data0")
    vars_key = states.var("vars0")
    mdata = MData(
        data={
            FC.STATE: np.arange(2),
            vars_key: np.array([FV.WS]),
            data_key: np.array([[8.0], [9.0], [10.0]]),
        },
        dims={
            FC.STATE: (FC.STATE,),
            vars_key: (vars_key,),
            data_key: (FC.TURBINE, vars_key),
        },
        extra_data={states.META: {"data_keys": [data_key]}},
    )
    fdata = FData(
        data={FV.ORDER: np.array([[2, 0, 1], [1, 2, 0]])},
        dims={FV.ORDER: (FC.STATE, FC.TURBINE)},
    )

    data, _ = states._get_calc_data(mdata, fdata)

    variables, values = data[(FC.STATE, FC.TURBINE, vars_key)]
    assert variables == [FV.WS]
    np.testing.assert_allclose(
        values[..., 0],
        [[10.0, 8.0, 9.0], [9.0, 10.0, 8.0]],
    )
