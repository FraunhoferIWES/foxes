"""Tests for state-turbine performance evaluation."""

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.algorithms.downwind.models.prepare_for_output import PrepareForOutput
from foxes.core import FData, MData, OutputList
from foxes.models.outputs import FarmResultsChunkWriter, FarmResultsOutput
from foxes.models.turbine_models import TurbineEvaluation


METRICS = [
    FV.CAP,
    FV.EFF,
    FV.CAPF,
    FV.YLD,
    FV.P75,
    FV.P90,
    FV.AMB_CAPF,
    FV.AMB_YLD,
    FV.AMB_P75,
    FV.AMB_P90,
]


def _case(ordered=False):
    """Provide synthetic powers, mixed units and independently known metrics."""
    # Distinct units expose accidental assumptions about a common power unit.
    algo = SimpleNamespace(
        farm_controller=SimpleNamespace(
            turbine_types=[
                SimpleNamespace(P_nominal=1000.0, P_unit="kW"),
                SimpleNamespace(P_nominal=2.0, P_unit="MW"),
                SimpleNamespace(P_nominal=3.0e6, P_unit="W"),
            ]
        )
    )
    power = np.array([[1000.0, 1.0, 1.5e6], [250.0, 2.0, 3.0e6], [500.0, 0.0, 1.5e6]])
    capacity = np.broadcast_to([1000.0, 2.0, 3.0e6], power.shape)
    ambient = capacity.copy()

    # Literal yields make the expected conversion independent of model code.
    annual_yield = np.array(
        [[8.76, 8.76, 13.14], [2.19, 17.52, 26.28], [4.38, 0.0, 13.14]]
    )
    ambient_yield = np.broadcast_to([8.76, 17.52, 26.28], power.shape)
    expected = {
        FV.CAP: capacity,
        FV.EFF: power / ambient,
        FV.CAPF: power / capacity,
        FV.AMB_CAPF: ambient / capacity,
        FV.YLD: annual_yield,
        FV.AMB_YLD: ambient_yield,
        FV.P75: annual_yield * 0.9325,
        FV.P90: annual_yield * 0.8718,
        FV.AMB_P75: ambient_yield * 0.9325,
        FV.AMB_P90: ambient_yield * 0.8718,
    }

    # Reorder inputs and independent expectations by original turbine identity.
    data = {FV.P: power, FV.AMB_P: ambient, FV.WEIGHT: np.full_like(power, 0.1)}
    if ordered:
        order = np.array([[2, 0, 1], [1, 2, 0], [0, 2, 1]])
        states = np.arange(3)[:, None]
        data = {v: values[states, order] for v, values in data.items()}
        expected = {v: values[states, order] for v, values in expected.items()}
        data[FV.ORDER] = order
    fdata = FData(data=data, dims={v: (FC.STATE, FC.TURBINE) for v in data})
    return algo, fdata, expected


def test_supported_metrics_match_public_metadata():
    assert TurbineEvaluation.SUPPORTED_VARIABLES == set(METRICS)
    assert TurbineEvaluation.BASE_VARIABLES == {
        FV.amb2var.get(variable, variable) for variable in METRICS
    }
    for variable in METRICS:
        assert getattr(FV, variable) == variable
        if variable in FV.amb2var:
            assert FV.var2amb[FV.amb2var[variable]] == variable


def test_evaluation_is_not_registered_as_an_output():
    for name in ("FarmResultsEval", "WindFarmsEval", "ClusterEval"):
        assert not hasattr(foxes.models.outputs, name)
        assert not hasattr(foxes, "output")
    with pytest.raises(KeyError):
        foxes.ModelBook().outputs["farm_results_eval"]


@pytest.fixture(params=[foxes.algorithms.Downwind, foxes.algorithms.Iterative])
def output_algorithm(request):
    farm = foxes.WindFarm()
    for i, ttype in enumerate(["NREL5MW", "DTU10MW"]):
        farm.add_turbine(
            foxes.Turbine([700.0 * i, 0.0], turbine_models=[ttype]), verbosity=0
        )
    states = foxes.input.states.StatesTable(
        pd.DataFrame({FV.WD: [90.0, 270.0], FV.WEIGHT: [0.25, 0.75]}),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WS: 8.0, FV.TI: 0.08, FV.RHO: 1.225},
    )
    return request.param(
        farm,
        states,
        rotor_model="centre",
        wake_models=["Bastankhah2014_linear_lim_k005"],
        verbosity=0,
    )


@pytest.mark.parametrize("engine", ["numpy", "threads", "process"])
@pytest.mark.parametrize("ambient", [False, True])
def test_output_selection_automatically_evaluates_metrics(
    output_algorithm, engine, ambient
):
    algo = output_algorithm
    variables = [
        FV.P,
        FV.AMB_P,
        FV.CAP,
        FV.CAPF,
        FV.AMB_CAPF,
        FV.EFF,
        FV.YLD,
        FV.AMB_YLD,
    ]
    with foxes.Engine.new(engine, n_procs=2, chunk_size_states=1, verbosity=0):
        results = algo.calc_farm(
            ambient=ambient,
            outputs=FarmResultsOutput(variables=variables),
        )
    capacity = np.broadcast_to([5000.0, 10000.0], (2, 2))
    np.testing.assert_allclose(results[FV.CAP], capacity)
    np.testing.assert_allclose(results[FV.EFF], results[FV.P] / results[FV.AMB_P])
    for power, factor, annual_yield in (
        (FV.P, FV.CAPF, FV.YLD),
        (FV.AMB_P, FV.AMB_CAPF, FV.AMB_YLD),
    ):
        np.testing.assert_allclose(results[factor], results[power] / capacity)
        np.testing.assert_allclose(results[annual_yield], results[power] * 0.00876)
    for variable in variables:
        assert results[variable].dims == (FC.STATE, FC.TURBINE)
    assert not any(
        isinstance(model, TurbineEvaluation)
        for model in algo.mbook.turbine_models.values()
    )


def test_unrestricted_output_keeps_only_available_fields(output_algorithm):
    with foxes.Engine.new("numpy", chunk_size_states=1, verbosity=0):
        results = output_algorithm.calc_farm(outputs="farm_results_all")
    assert FV.P in results and FV.AMB_P in results
    assert not TurbineEvaluation.SUPPORTED_VARIABLES.intersection(results.data_vars)


@pytest.mark.parametrize("reorder", [False, True])
def test_output_preparation_evaluates_before_restoring_order(reorder):
    algo, fdata, ordered_expected = _case(ordered=True)
    _, _, original_expected = _case()
    variables = [FV.CAP, FV.CAPF, FV.EFF, FV.YLD, FV.AMB_CAPF, FV.AMB_YLD]
    algo.farm_vars = [FV.P, FV.AMB_P]
    model = PrepareForOutput(
        output=FarmResultsOutput(variables=variables), reorder=reorder
    )
    model.initialize(algo, verbosity=0)

    model.calculate(algo, MData(), fdata)

    expected = original_expected if reorder else ordered_expected
    for variable in variables:
        np.testing.assert_allclose(fdata[variable], expected[variable])
    assert FV.ORDER not in fdata
    assert FV.P not in fdata
    model.finalize(algo, verbosity=0)


@pytest.fixture(params=[foxes.algorithms.Downwind, foxes.algorithms.Iterative])
def mixed_unit_order_algorithm(request):
    """Build distinct turbine identities with a non-involutive downwind order."""
    mbook = foxes.ModelBook()
    farm = foxes.WindFarm()
    metadata = [(1000.0, "kW"), (2.0, "MW"), (3.0e6, "W")]
    positions = [[0.0, 0.0], [800.0, 200.0], [250.0, 1100.0]]
    for i, ((capacity, unit), position) in enumerate(zip(metadata, positions)):
        name = f"mixed_unit_{i}"
        mbook.turbine_types[name] = foxes.models.turbine_types.PCtFile(
            pd.DataFrame(
                {
                    "ws": [0.0, 8.0, 30.0],
                    "P": [0.0, capacity / 2, capacity],
                    "ct": [0.0, 0.8, 0.8],
                }
            ),
            name=name,
            D=100.0,
            H=100.0,
            P_nominal=capacity,
            P_unit=unit,
        )
        farm.add_turbine(foxes.Turbine(position, turbine_models=[name]), verbosity=0)
    states = foxes.input.states.StatesTable(
        pd.DataFrame({FV.WD: [90.0, 180.0, 270.0]}),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WS: 8.0, FV.TI: 0.08, FV.RHO: 1.225},
    )
    return request.param(
        farm,
        states,
        mbook=mbook,
        rotor_model="centre",
        wake_models=["Jensen_linear_k004"],
        verbosity=0,
    )


@pytest.mark.parametrize("engine", ["numpy", "threads", "process"])
@pytest.mark.parametrize("chunk_size", [1, 2])
def test_output_metrics_follow_original_identity_across_three_turbine_orders(
    mixed_unit_order_algorithm, engine, chunk_size
):
    """Keep normalized public metrics aligned with native turbine identities."""
    variables = [
        FV.P,
        FV.AMB_P,
        FV.CAP,
        FV.CAPF,
        FV.AMB_CAPF,
        FV.EFF,
        FV.YLD,
        FV.AMB_YLD,
        FV.ORDER,
        FV.X,
        FV.Y,
    ]
    with foxes.Engine.new(engine, n_procs=2, chunk_size_states=chunk_size, verbosity=0):
        results = mixed_unit_order_algorithm.calc_farm(
            outputs=FarmResultsOutput(variables=variables)
        )

    np.testing.assert_array_equal(results[FV.ORDER], [[1, 2, 0], [0, 1, 2], [0, 2, 1]])
    if isinstance(mixed_unit_order_algorithm, foxes.algorithms.Iterative):
        downwind_results = mixed_unit_order_algorithm.farm_results_downwind
        native_ambient = np.array([500.0, 1.0, 1.5e6])[
            downwind_results[FV.ORDER].to_numpy()
        ]
        np.testing.assert_allclose(downwind_results[FV.AMB_P], native_ambient)

    np.testing.assert_allclose(results[FV.CAP], np.tile([1.0e6, 2.0e6, 3.0e6], (3, 1)))
    assert results[FV.P].attrs["units"] == results[FV.CAP].attrs["units"] == "W"
    np.testing.assert_allclose(results[FV.X], np.tile([0.0, 800.0, 250.0], (3, 1)))
    np.testing.assert_allclose(results[FV.Y], np.tile([0.0, 200.0, 1100.0], (3, 1)))
    for power, factor, annual_yield in (
        (FV.P, FV.CAPF, FV.YLD),
        (FV.AMB_P, FV.AMB_CAPF, FV.AMB_YLD),
    ):
        np.testing.assert_allclose(
            results[factor], results[power] / [1.0e6, 2.0e6, 3.0e6]
        )
        np.testing.assert_allclose(
            results[annual_yield],
            results[power] * 8760 / 1.0e9,
        )


@pytest.mark.parametrize("variable", [FV.P75, FV.AMB_P90])
def test_output_selected_probability_yields_require_uncertainty(
    output_algorithm, variable
):
    with foxes.Engine.new("numpy", verbosity=0):
        with pytest.raises(ValueError, match="power_uncert is required"):
            output_algorithm.calc_farm(outputs=FarmResultsOutput(variables=[variable]))


def test_mixed_unit_outputs_can_be_reprocessed_and_aggregated(
    mixed_unit_order_algorithm, tmp_path
):
    """Reprocess normalized power without converting it again before grouping."""
    algo = mixed_unit_order_algorithm
    fields = [FV.P, FV.AMB_P, FV.CAP, FV.YLD, FV.EFF, FV.CAPF, FV.WEIGHT]
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        original = algo.calc_farm(outputs=FarmResultsOutput(variables=fields))
        outputs = [
            FarmResultsOutput(variables=fields),
            FarmResultsOutput(
                variables=fields, aggregate_turbines="all", aggregate_states=True
            ),
            FarmResultsChunkWriter(
                variables=fields,
                aggregate_turbines="all",
                base_name="mixed",
                out_dir=tmp_path,
            ),
        ]
        results = algo.calc_outputs(farm_results=original, outputs=outputs)

    xr.testing.assert_identical(results[FC.FARM_RESULTS], original)
    grouped = results[FC.MEAN_TOTAL_FARM_RESULTS]
    expected = original[FV.P].mean(FC.STATE).sum(FC.TURBINE)
    np.testing.assert_allclose(grouped[FV.P], expected)
    np.testing.assert_allclose(grouped[FV.CAP], 6e6)
    np.testing.assert_allclose(grouped[FV.CAPF], expected / 6e6)
    np.testing.assert_allclose(grouped[FV.YLD], expected * 8760 / 1e9)
    assert grouped[FV.P].attrs["units"] == "W"

    for path in tmp_path.glob("mixed_*.nc"):
        with xr.open_dataset(path) as written:
            assert written[FV.P].attrs["units"] == written[FV.CAP].attrs["units"] == "W"
            np.testing.assert_allclose(written[FV.CAP], 6e6)


@pytest.mark.parametrize("nested", [False, True])
def test_explicit_metrics_survive_unrestricted_output_composition(
    output_algorithm, tmp_path, nested
):
    collector = FarmResultsOutput()
    writer = FarmResultsChunkWriter(
        variables=[FV.EFF], base_name="metrics", out_dir=tmp_path
    )
    outputs = [collector, OutputList(outputs=[writer]) if nested else writer]
    with foxes.Engine.new("numpy", chunk_size_states=1, verbosity=0):
        results = output_algorithm.calc_farm(outputs=outputs)[FC.FARM_RESULTS]
    np.testing.assert_allclose(results[FV.EFF], results[FV.P] / results[FV.AMB_P])
    paths = sorted(tmp_path.glob("metrics*.nc"))
    assert len(paths) == 2
    for i, path in enumerate(paths):
        with xr.open_dataset(path) as written:
            np.testing.assert_allclose(
                written[FV.EFF], results[FV.EFF].isel({FC.STATE: slice(i, i + 1)})
            )


def test_efficiency_returns_full_selected_farm_array():
    model = TurbineEvaluation([FV.EFF])
    model.initialize(algo=None, verbosity=0)
    data = {
        FV.P: np.array([[1.0, 3.0], [2.0, 4.0]]),
        FV.AMB_P: np.full((2, 2), 4.0),
        FV.EFF: np.full((2, 2), -1.0),
    }
    fdata = FData(data=data, dims={v: (FC.STATE, FC.TURBINE) for v in data})
    selection = np.array([[True, False], [False, True]])

    results = model.calculate(None, MData(), fdata, selection)

    np.testing.assert_array_equal(results[FV.EFF], [[0.25, -1.0], [-1.0, 1.0]])
    assert results[FV.EFF] is fdata[FV.EFF]


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize(
    "selection",
    [
        slice(None),
        slice(1, 3),
        slice(None, None, -1),
        np.array([[True, False, True], [False, True, False], [True, False, False]]),
        np.s_[:, 1],
        np.s_[np.array([False, True, False]), 2],
        np.zeros((3, 3), dtype=bool),
    ],
)
def test_metrics_preserve_shape_selection_order_and_running_model(ordered, selection):
    # Capture the running instance to detect attribute or configuration mutation.
    algo, fdata, expected = _case(ordered)
    model = TurbineEvaluation(METRICS, power_uncert=0.1)
    model.initialize(algo, verbosity=0)
    model.ensure_output_vars(algo, fdata)
    for variable in METRICS:
        fdata[variable][:] = -7.0
    model.set_running(algo, {})
    serialized = pickle.dumps(model)
    original_power = fdata[FV.P].copy()

    results = model.calculate(algo, MData(), fdata, selection)

    # Preserve prefilled output values outside each NumPy selection.
    for variable in METRICS:
        values = np.full((3, 3), -7.0)
        values[selection] = expected[variable][selection]
        np.testing.assert_allclose(results[variable], values)
        assert results[variable] is fdata[variable]
        assert fdata.dims[variable] == (FC.STATE, FC.TURBINE)
    np.testing.assert_array_equal(fdata[FV.P], original_power)
    assert pickle.dumps(model) == serialized
    model.unset_running(algo, {})
    model.finalize(algo, verbosity=0)


def test_call_time_variables_add_full_arrays_without_changing_configuration():
    algo, fdata, expected = _case()
    configured = [FV.YLD, FV.YLD]
    model = TurbineEvaluation(configured, name="metrics", power_uncert=0.1)
    model.initialize(algo, verbosity=0)
    configured.append(FV.CAP)

    results = model.calculate(algo, MData(), fdata, np.s_[:, 1], variables=METRICS)

    assert model.name == "metrics"
    assert model.output_farm_vars(algo) == [FV.YLD]
    assert set(results) == set(METRICS)
    for variable, values in results.items():
        np.testing.assert_allclose(values[:, 1], expected[variable][:, 1])
        assert np.isnan(values[:, [0, 2]]).all()


def test_empty_variables_request_does_not_change_farm_data():
    algo, fdata, _ = _case()
    model = TurbineEvaluation()
    model.initialize(algo, verbosity=0)
    keys = set(fdata)

    assert model.calculate(algo, MData(), fdata) == {}
    assert set(fdata) == keys


@pytest.mark.parametrize("call_time", [False, True])
def test_unknown_metrics_are_rejected(call_time):
    with pytest.raises(ValueError, match="Unsupported evaluation variables.*unknown"):
        if call_time:
            algo, fdata, _ = _case()
            model = TurbineEvaluation()
            model.initialize(algo, verbosity=0)
            model.calculate(algo, MData(), fdata, variables=["unknown"])
        else:
            TurbineEvaluation(["unknown"])


@pytest.mark.parametrize("variable", [FV.P75, FV.P90, FV.AMB_P75, FV.AMB_P90])
@pytest.mark.parametrize("call_time", [False, True])
def test_probability_yields_require_uncertainty(variable, call_time):
    with pytest.raises(ValueError, match="power_uncert is required"):
        if call_time:
            algo, fdata, _ = _case()
            model = TurbineEvaluation()
            model.initialize(algo, verbosity=0)
            model.calculate(algo, MData(), fdata, variables=[variable])
        else:
            TurbineEvaluation([variable])


@pytest.mark.parametrize("uncertainty", [-0.1, np.nan, np.inf, -np.inf])
def test_invalid_uncertainty_is_rejected(uncertainty):
    with pytest.raises(ValueError, match="power_uncert must be finite and nonnegative"):
        TurbineEvaluation([FV.P75], power_uncert=uncertainty)


def test_zero_uncertainty_and_unweighted_yield():
    algo, fdata, expected = _case()
    fdata[FV.WEIGHT][:] = 0.0
    model = TurbineEvaluation([FV.YLD, FV.P75, FV.P90], power_uncert=0.0)
    model.initialize(algo, verbosity=0)

    results = model.calculate(algo, MData(), fdata)

    for values in results.values():
        np.testing.assert_allclose(values, expected[FV.YLD])


@pytest.mark.parametrize("nominal", [None, 0.0, -1.0, np.nan, np.inf])
def test_invalid_nominal_power_is_rejected(nominal):
    algo, fdata, _ = _case()
    algo.farm_controller.turbine_types[0].P_nominal = nominal
    model = TurbineEvaluation([FV.CAPF])
    model.initialize(algo, verbosity=0)

    with pytest.raises(ValueError, match="P_nominal must be finite and positive"):
        model.calculate(algo, MData(), fdata)


def test_invalid_unselected_nominal_power_is_not_used():
    algo, fdata, _ = _case()
    algo.farm_controller.turbine_types[0].P_nominal = np.nan
    model = TurbineEvaluation([FV.CAP])
    model.initialize(algo, verbosity=0)

    results = model.calculate(algo, MData(), fdata, np.s_[:, 1])

    np.testing.assert_array_equal(results[FV.CAP][:, 1], 2.0)


def test_unsupported_power_unit_is_contextual():
    algo, fdata, _ = _case()
    algo.farm_controller.turbine_types[0].P_unit = "unknown"
    model = TurbineEvaluation([FV.YLD])
    model.initialize(algo, verbosity=0)

    with pytest.raises(ValueError, match="Unsupported turbine power unit 'unknown'"):
        model.calculate(algo, MData(), fdata)


def test_missing_turbine_types_is_contextual():
    algo, fdata, _ = _case()
    algo.farm_controller.turbine_types = None
    model = TurbineEvaluation([FV.CAP])
    model.initialize(algo, verbosity=0)

    with pytest.raises(ValueError, match="turbine types are unavailable"):
        model.calculate(algo, MData(), fdata)


@pytest.mark.parametrize("variable,missing", [(FV.YLD, FV.P), (FV.EFF, FV.AMB_P)])
def test_missing_power_field_is_contextual(variable, missing):
    algo, fdata, _ = _case()
    del fdata[missing], fdata.dims[missing]
    model = TurbineEvaluation([variable])
    model.initialize(algo, verbosity=0)

    with pytest.raises(ValueError, match=f"Variable '{missing}'.*not found"):
        model.calculate(algo, MData(), fdata)


@pytest.mark.parametrize(
    "selection",
    [
        slice(None),
        slice(1, 3),
        np.array([[True, False, True, False], [False, True, True, False], [True] * 4]),
        np.s_[:, 0],
        np.s_[0, 0],
        np.zeros((3, 4), dtype=bool),
    ],
)
def test_efficiency_handles_zero_division_without_warnings(selection):
    data = {
        FV.P: np.array(
            [[0.0, 1.0, -1.0, 2.0], [0.0, -0.0, np.nan, np.nan], [0.0, 2.0, 0.0, 0.0]]
        ),
        FV.AMB_P: np.array(
            [
                [0.0, 0.0, -0.0, 1.0],
                [5.0, -2.0, 0.0, 1.0],
                [np.nan, np.nan, np.inf, -np.inf],
            ]
        ),
        FV.EFF: np.full((3, 4), -1.0),
    }
    reference = np.array(
        [
            [0.0, np.inf, np.inf, 2.0],
            [0.0, 0.0, np.nan, np.nan],
            [0.0, np.nan, 0.0, 0.0],
        ]
    )
    expected = data[FV.EFF].copy()
    expected[selection] = reference[selection]
    fdata = FData(data=data, dims={v: (FC.STATE, FC.TURBINE) for v in data})
    model = TurbineEvaluation([FV.EFF])
    model.initialize(None, verbosity=0)

    with np.errstate(divide="raise", invalid="raise"):
        results = model.calculate(None, MData(), fdata, selection)

    np.testing.assert_array_equal(results[FV.EFF], expected)
    assert results[FV.EFF] is fdata[FV.EFF]


def test_efficiency_accepts_integer_power_fields():
    data = {
        FV.P: np.array([[0, -2, 4]]),
        FV.AMB_P: np.array([[0, 0, 2]]),
    }
    fdata = FData(data=data, dims={v: (FC.STATE, FC.TURBINE) for v in data})
    model = TurbineEvaluation([FV.EFF])
    model.initialize(None, verbosity=0)

    with np.errstate(divide="raise", invalid="raise"):
        results = model.calculate(None, MData(), fdata)

    np.testing.assert_array_equal(results[FV.EFF], [[0.0, np.inf, 2.0]])


@pytest.mark.parametrize("engine", ["numpy", "threads", "process"])
def test_zero_power_ratios_are_automatically_evaluated(output_algorithm, engine):
    algo = output_algorithm
    algo.states.fixed_vars[FV.WS] = 0.0
    variables = [FV.P, FV.AMB_P, FV.EFF, FV.CAPF, FV.AMB_CAPF]
    with foxes.Engine.new(engine, n_procs=2, chunk_size_states=1, verbosity=0):
        results = algo.calc_farm(outputs=FarmResultsOutput(variables=variables))

    for variable in variables:
        np.testing.assert_array_equal(results[variable], np.zeros((2, 2)))


def test_probability_variable_metadata():
    for variable in [FV.P75, FV.P90]:
        ambient = FV.var2amb[variable]
        assert FV.amb2var[ambient] == variable
        assert {variable, ambient} <= FV.extensive_farm
        assert {variable, ambient} <= FV.extensive_state
        assert not {variable, ambient}.intersection(FV.intensive_farm)
        assert not {variable, ambient}.intersection(FV.intensive_state)
        assert FV.get_default_digits(variable) == FV.get_default_digits(ambient) == 3


@pytest.mark.parametrize(
    "algorithm,relaxation",
    [
        (foxes.algorithms.Downwind, False),
        (foxes.algorithms.Iterative, False),
        (foxes.algorithms.Iterative, True),
    ],
)
@pytest.mark.parametrize(
    "engine,chunk_size", [("numpy", 3), ("single", 1), ("threads", 1), ("process", 1)]
)
@pytest.mark.parametrize("selected_only", [False, True])
def test_real_farm_metrics_match_power_across_chunks_and_engines(
    algorithm, relaxation, engine, chunk_size, selected_only
):
    mbook = foxes.ModelBook()
    mbook.turbine_models["metrics"] = TurbineEvaluation(METRICS, power_uncert=0.1)
    farm = foxes.WindFarm()
    for index, turbine_type in enumerate(["NREL5MW", "DTU10MW", "NREL5MW"]):
        farm.add_turbine(
            foxes.Turbine(
                xy=[700.0 * index, 0.0],
                turbine_models=[turbine_type]
                + (["metrics"] if not selected_only or index == 1 else []),
            ),
            verbosity=0,
        )
    states = foxes.input.states.StatesTable(
        pd.DataFrame({FV.WD: [90.0, 270.0, 180.0], FV.WEIGHT: [0.2, 0.3, 0.5]}),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WS: 8.0, FV.TI: 0.08, FV.RHO: 1.225},
    )
    algo = algorithm(
        farm,
        states,
        mbook=mbook,
        rotor_model="centre",
        wake_models=["Bastankhah2014_linear_lim_k005"],
        verbosity=0,
    )
    if relaxation:
        algo.max_it = 2
        algo.set_urelax("last", P=0.5)
    with foxes.Engine.new(engine, n_procs=2, chunk_size_states=chunk_size, verbosity=0):
        results = algo.calc_farm(outputs="farm_results_all")

    capacity = np.broadcast_to(
        [mbook.turbine_types[t].P_nominal for t in ["NREL5MW", "DTU10MW", "NREL5MW"]],
        (3, 3),
    )
    selected = np.ones((3, 3), dtype=bool)
    if selected_only:
        selected[:, [0, 2]] = False
    expected_capacity = np.where(selected, capacity, np.nan)
    expected_efficiency = np.where(selected, results[FV.P] / results[FV.AMB_P], np.nan)
    np.testing.assert_allclose(results[FV.CAP], expected_capacity)
    np.testing.assert_allclose(results[FV.EFF], expected_efficiency)
    for ambient in [False, True]:
        power_var = FV.AMB_P if ambient else FV.P
        power = results[power_var].to_numpy()
        for metric, expected in {
            FV.CAPF: power / capacity,
            FV.YLD: power * 0.00876,
            FV.P75: power * 0.00876 * 0.9325,
            FV.P90: power * 0.00876 * 0.8718,
        }.items():
            output_var = FV.var2amb[metric] if ambient else metric
            assert results[output_var].dims == (FC.STATE, FC.TURBINE)
            np.testing.assert_allclose(
                results[output_var], np.where(selected, expected, np.nan)
            )


@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize("with_wakes", [False, True])
def test_ambient_only_metrics_and_single_turbine_need_final_evaluation(
    ambient, with_wakes
):
    variables = [FV.EFF, FV.AMB_CAPF, FV.AMB_YLD, FV.AMB_P75, FV.AMB_P90]
    mbook = foxes.ModelBook()
    mbook.turbine_models["metrics"] = TurbineEvaluation(variables, power_uncert=0.1)
    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(xy=[0.0, 0.0], turbine_models=["NREL5MW", "metrics"]),
        verbosity=0,
    )
    states = foxes.input.states.StatesTable(
        pd.DataFrame({FV.WD: [90.0]}),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WS: 8.0, FV.TI: 0.08, FV.RHO: 1.225},
    )
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        mbook=mbook,
        rotor_model="centre",
        wake_models=["Bastankhah2014_linear_lim_k005"] if with_wakes else [],
        verbosity=0,
    )
    with foxes.Engine.new("numpy", verbosity=0):
        results = algo.calc_farm(
            ambient=ambient,
            outputs=FarmResultsOutput(variables=[FV.AMB_P, *variables]),
        )

    power = results[FV.AMB_P].to_numpy()
    np.testing.assert_allclose(results[FV.EFF], 1.0)
    for metric, expected in {
        FV.AMB_CAPF: power / mbook.turbine_types["NREL5MW"].P_nominal,
        FV.AMB_YLD: power * 0.00876,
        FV.AMB_P75: power * 0.00876 * 0.9325,
        FV.AMB_P90: power * 0.00876 * 0.8718,
    }.items():
        np.testing.assert_allclose(results[metric], expected)
