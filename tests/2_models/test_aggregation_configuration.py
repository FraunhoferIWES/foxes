"""Aggregation preset coverage, provenance and YAML execution."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config
from foxes.input.yaml.dict import run_outputs
from foxes.models import outputs
from foxes.models._model_book_outputs import _farm_variable_sets, _point_variable_sets
from foxes.models.outputs._result_data import _Results, _combine_chunks
from foxes.utils import Dict


@pytest.fixture(scope="module")
def book():
    return foxes.ModelBook()


@pytest.fixture
def algorithm():
    states = foxes.input.states.StatesTable(
        pd.DataFrame(
            {
                FV.WS: [6.0, 8.0, 10.0, 12.0, 14.0],
                FV.WEIGHT: [0.05, 0.15, 0.2, 0.25, 0.35],
            },
            index=pd.date_range("2020", periods=5, freq="12h"),
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WD: 270.0, FV.TI: 0.08, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for i in range(3):
        farm.add_turbine(
            foxes.Turbine([i * 500.0, 0.0], turbine_models=["NREL5MW"]),
            verbosity=0,
        )
    return foxes.algorithms.Downwind(farm, states, wake_models=[], verbosity=0)


def _point_data(states, spacing=1.0):
    """Provide deterministic point fields with unevenly chunked target coordinates."""
    return xr.Dataset(
        {
            FV.WS: ((FC.STATE, FC.POINT), np.arange(25.0).reshape(5, 5)),
            FV.WEIGHT: ((FC.STATE, FC.POINT), np.full((5, 5), 0.2)),
        },
        coords={
            FC.STATE: states,
            FC.POINT: np.arange(5),
            FV.X: (FC.POINT, np.arange(5.0) * spacing),
            FV.Y: (FC.POINT, np.zeros(5)),
            FV.H: (FC.POINT, np.full(5, 90.0)),
        },
    )


@pytest.mark.parametrize(
    "domain,group,fields",
    [
        *(
            ("farm", group, fields)
            for group in (None, "wind_farm", "cluster", "all")
            for fields in ("", *(f"_{name}" for name in _farm_variable_sets()))
        ),
        *(
            ("point", group, fields)
            for group in (None, "all")
            for fields in ("", *(f"_{name}" for name in _point_variable_sets()))
        ),
    ],
)
@pytest.mark.parametrize(
    "filtering,ambient,waked",
    [
        ("", True, True),
        ("_amb", True, False),
        ("_waked", False, True),
    ],
)
@pytest.mark.parametrize("mean", [False, True])
@pytest.mark.parametrize(
    "policy",
    [
        "",
        "_write_n2",
        "_write_chunks",
        "_write_day",
        "_write_week",
        "_write_month",
        "_write_year",
    ],
)
def test_complete_preset_matrix(
    book, domain, group, filtering, ambient, waked, fields, mean, policy
):
    prefix = {
        ("farm", "wind_farm"): "farms_results",
        ("farm", "cluster"): "cluster_results",
        ("farm", "all"): "total_results",
    }.get((domain, group), f"{domain}_results")
    name = prefix + filtering + fields
    name += "_aggregate_all" if domain == "point" and group == "all" else ""
    name += "_mean" if mean else ""
    output = book.outputs[name + policy]
    family = domain.title()
    kind = (
        "Output"
        if not policy
        else "CountWriter"
        if policy == "_write_n2"
        else "ChunkWriter"
        if policy == "_write_chunks"
        else "PeriodWriter"
    )
    assert type(output) is getattr(outputs, f"{family}Results{kind}")
    parameter = "aggregate_turbines" if domain == "farm" else "aggregate_points"
    assert getattr(output, parameter) == group
    assert output.aggregate_states is mean
    assert output.ambient_vars is ambient
    assert output.waked_vars is waked
    if policy:
        assert output.base_name == name
    if policy == "_write_n2":
        assert output.n_states_per_file == 2
    elif policy not in ("", "_write_chunks"):
        assert output.period == policy.removeprefix("_write_")


@pytest.mark.parametrize("policy", ["", "_write_n2", "_write_chunks", "_write_month"])
def test_named_farm_analysis_combines_filter_group_and_writer(book, policy):
    name = "cluster_results_amb_analysis_mean"
    output = book.outputs[name + policy]
    assert output.aggregate_turbines == "cluster"
    assert output.aggregate_states
    assert output.ambient_vars and not output.waked_vars
    if policy:
        assert output.base_name == name


@pytest.mark.parametrize(
    "name",
    [
        "farm_results_aggregate_wind_farm",
        "farm_results_aggregate_cluster_mean",
        "farm_results_amb_analysis_aggregate_cluster_mean_write_month",
        "farms_results_aggregate_wind_farm",
        "cluster_results_aggregate_cluster",
        "farm_results_aggregate_all",
        "farm_results_all_aggregate_all_mean",
        "farm_results_amb_analysis_aggregate_all_mean_write_month",
        "total_results_aggregate_all",
        "total_results_all_aggregate_all_mean_write_n2",
        "farms_results_aggregate_all",
        "cluster_results_aggregate_all",
    ],
)
def test_grouped_presets_do_not_keep_redundant_or_previous_names(book, name):
    with pytest.raises(KeyError):
        book.outputs[name]


@pytest.mark.parametrize(
    "domain,group", [("farm", None), ("farm", "all"), ("point", None), ("point", "all")]
)
@pytest.mark.parametrize("mean", [False, True])
def test_counts_describe_only_eliminated_dimensions(algorithm, domain, group, mean):
    parameter = "aggregate_turbines" if domain == "farm" else "aggregate_points"
    cls = outputs.FarmResultsOutput if domain == "farm" else outputs.PointResultsOutput
    variable = FV.P if domain == "farm" else FV.WS
    output = cls(variables=[variable], **{parameter: group}, aggregate_states=mean)
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        raw = algorithm.calc_farm(outputs="farm_results_all")
        kwargs = {"farm_results": raw}
        if domain == "point":
            kwargs = {"point_results": _point_data(raw[FC.STATE], spacing=10)}
        result = algorithm.calc_outputs(
            outputs=output, states_sel=raw[FC.STATE].values[[4, 1, 3]], **kwargs
        )
    assert result.attrs.get("n_states") == (3 if mean else None)
    count = "n_turbines" if domain == "farm" else "n_points"
    assert result.attrs.get(count) == (
        (3 if domain == "farm" else 5) if group else None
    )
    assert not any("weight" in key or key.startswith("__") for key in result.attrs)
    assert set(result.data_vars) - {FC.TNAME} == {variable}


@pytest.mark.parametrize("domain", ["total", "farms", "cluster", "point"])
@pytest.mark.parametrize("policy", ["", "_write_n2", "_write_chunks", "_write_day"])
def test_aggregation_presets_execute_and_serialize_counts(
    algorithm, tmp_path, monkeypatch, domain, policy
):
    monkeypatch.setitem(config, FC.OUTPUT_DIR, tmp_path)
    aggregation = "_aggregate_all" if domain == "point" else ""
    name = f"{domain}_results_waked_all{aggregation}_mean{policy}"
    output = algorithm.mbook.outputs[name]
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ) as engine:
        state_sizes, target_sizes = engine.calc_chunk_sizes(5, 5)
        if domain != "point":
            result = algorithm.calc_farm(outputs=name)
        else:
            data = _point_data(pd.date_range("2020", periods=5, freq="12h"))
            result = algorithm.calc_outputs(outputs=name, point_results=data)
    if not policy:
        assert result.attrs["n_states"] == 5
        assert result.attrs["n_points" if domain == "point" else "n_turbines"] == (
            5 if domain == "point" else 3
        )
        return
    assert result == {}
    paths = sorted(tmp_path.glob(f"{output.base_name}*.nc"))
    target_chunks = len(target_sizes) if domain == "point" else 1
    expected_files = (
        len(state_sizes) * target_chunks if policy == "_write_chunks" else 3
    )
    assert len(paths) == expected_files
    counts = []
    for path in paths:
        with xr.open_dataset(path) as data:
            counts.append(data.attrs["n_states"])
            count = data.attrs["n_points" if domain == "point" else "n_turbines"]
            if domain == "point" and policy == "_write_chunks":
                assert count in target_sizes
            else:
                assert count == (5 if domain == "point" else 3)
            assert FC.STATE not in data.dims
    assert sum(counts) == (5 * target_chunks if policy == "_write_chunks" else 5)


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize(
    "group,selector",
    [
        ("all", "total_results_all_mean"),
        ("wind_farm", "farms_results_all_mean"),
        ("cluster", "cluster_results_all_mean"),
    ],
)
def test_yaml_executes_aggregated_outputs_with_preselected_states(
    algorithm,
    named,
    group,
    selector,
):
    if named:
        algorithm.mbook.outputs[selector].nofig = False
    definition = Dict(
        outputs=[
            Dict(
                output_type=selector if named else "FarmResultsOutput",
                states_isel=[4, 1, 3],
                result_labels="$summary",
                **(
                    {}
                    if named
                    else {
                        "variables": [FV.P],
                        "aggregate_turbines": group,
                        "aggregate_states": True,
                    }
                ),
            )
        ]
    )
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        raw = algorithm.calc_farm(outputs="farm_results_all")
        _, stored = run_outputs(
            definition, algorithm, raw, with_engine=True, nofig=True, verbosity=0
        )
    result = stored["$summary"]
    selected = raw.isel({FC.STATE: [4, 1, 3]})
    weights = selected[FV.WEIGHT].broadcast_like(selected[FV.P])
    expected = ((selected[FV.P] * weights).sum(FC.STATE) / weights.sum(FC.STATE)).sum()
    assert result[FV.P].item() == pytest.approx(float(expected))
    assert result.attrs["n_states"] == 3
    assert result.attrs["n_turbines"] == 3
    if named:
        assert not algorithm.mbook.outputs[selector].nofig


@pytest.mark.parametrize("mean", [False, True])
def test_total_presets_keep_public_keys_and_compose_with_raw_results(algorithm, mean):
    suffix = "_mean" if mean else ""
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        result = algorithm.calc_farm(outputs=["farm_results", f"total_results{suffix}"])
    key = FC.MEAN_TOTAL_FARM_RESULTS if mean else FC.TOTAL_FARM_RESULTS
    assert set(result) == {FC.FARM_RESULTS, key}
    raw, total = result[FC.FARM_RESULTS], result[key]
    expected = raw[FV.P]
    if mean:
        weights = raw[FV.WEIGHT].broadcast_like(expected)
        expected = (expected * weights).sum(FC.STATE) / weights.sum(FC.STATE)
    xr.testing.assert_allclose(total[FV.P], expected.sum(FC.TURBINE))
    assert total.attrs["n_turbines"] == raw.sizes[FC.TURBINE]
    assert total.attrs.get("n_states") == (raw.sizes[FC.STATE] if mean else None)
    assert (FC.STATE in total.dims) is (not mean)
    assert FC.TURBINE not in total.dims


@pytest.mark.parametrize(
    "options,message",
    [
        ({"states_isel": [0], "states_sel": ["2020"]}, "positions or labels"),
        ({"states_isel": []}, "integer positions"),
        ({"states_isel": [True]}, "integer positions"),
        ({"states_isel": [0, True]}, "integer positions"),
        ({"states_isel": [5]}, "outside"),
        ({"aggregate_states": True}, "constructor overrides"),
    ],
)
def test_yaml_rejects_ambiguous_or_invalid_preset_configuration(
    algorithm, options, message
):
    definition = Dict(outputs=[Dict(output_type="farm_results_all_mean", **options)])
    with (
        foxes.Engine.new("numpy", verbosity=0),
        pytest.raises(ValueError, match=message),
    ):
        run_outputs(definition, algorithm, with_engine=True, verbosity=0)


def test_shared_chunk_join_preserves_static_fields_and_checks_schemas():
    def chunk(start: int, label: str = "same") -> _Results:
        return dict(
            coords={FC.TARGET: np.arange(start, start + 2), "kind": np.array([label])},
            data_vars={
                FV.WS: ((FC.TARGET,), np.array([1.0, 2.0])),
                "static": ((), np.array("unchanged")),
            },
            attrs={"n_states": 3},
        )

    first, second = chunk(0), chunk(2)
    result = _combine_chunks([first, second], FC.TARGET)
    np.testing.assert_array_equal(result["coords"][FC.TARGET], np.arange(4))
    assert result["data_vars"]["static"][1] == "unchanged"
    assert result["attrs"] == {"n_states": 3}
    with pytest.raises(ValueError, match="Coordinate"):
        _combine_chunks([first, chunk(2, "different")], FC.TARGET)
    second["data_vars"]["static"] = ((), np.array("changed"))
    with pytest.raises(ValueError, match="Static field"):
        _combine_chunks([first, second], FC.TARGET)
    assert first["data_vars"]["static"][1] == "unchanged"


@pytest.mark.parametrize(
    "cls", [outputs.FarmResultsOutput, outputs.PointResultsOutput, outputs.SliceData]
)
@pytest.mark.parametrize("value", ["false", 1, None])
def test_aggregation_flags_require_booleans(cls, value):
    with pytest.raises(ValueError, match="boolean"):
        cls(aggregate_states=value)


@pytest.mark.parametrize("selection", ["states_sel", "states_isel"])
def test_yaml_resolves_referenced_state_selection(algorithm, selection):
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        farm = algorithm.calc_farm(outputs="farm_results_all")
        requested = (
            list(farm[FC.STATE].values[[4, 1]])
            if selection == "states_sel"
            else [-1, 1]
        )
        definition = Dict(
            outputs=[
                Dict(
                    output_type="farm_results_all_mean",
                    result_labels="$summary",
                    **{selection: "$selected"},
                )
            ]
        )
        _, stored = run_outputs(
            definition,
            algorithm,
            farm,
            with_engine=True,
            verbosity=0,
            results_storage=Dict({"$selected": requested}),
        )
    assert stored["$summary"].attrs["n_states"] == 2
