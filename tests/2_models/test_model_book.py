"""Registry ownership, factory conversion and lookup after setup decomposition."""

import pytest

import foxes.variables as FV
from foxes.models import ModelBook
from foxes.models.outputs import FarmResultsOutput
from foxes.utils.factory import _parse_coefficient


_FACTORY_TOKENS = {
    "n": ["3", "5"],
    "n2": ["9", "16"],
    "step": ["10", "100"],
    "chunksize": ["2", "10"],
    "chunk": ["0", "1"],
    "beta": ["01", "0.1"],
    "dx": ["10", "100"],
    "dt": ["30s", "2min"],
    "height": ["100", "500"],
    "kTI": ["02", "0.2"],
    "k": ["004", "0.04"],
    "ka": ["02", "0.2"],
    "ambka": ["04", "0.4"],
    "kb": ["0001", "0.001"],
    "superposition": ["linear", "quadratic"],
}


def test_registries_and_presets_are_independent_between_books():
    first, second = ModelBook(), ModelBook()
    assert set(first.sources) == set(first.base_classes)
    for collection, registry in first.sources.items():
        other = second.sources[collection]
        assert registry is getattr(first, collection)
        assert registry is not other
        assert list(registry) == list(other)
        for name in registry:
            assert registry[name] is not other[name]
            assert registry[name].name == name
        for left, right in zip(registry.factories, other.factories):
            assert left is not right
            assert str(left) == str(right)

    first.outputs["farm_results"].fdata_variables.append("test_only")
    assert "test_only" not in second.outputs["farm_results"].fdata_variables


@pytest.mark.parametrize(
    "collection, name, count, reduce",
    [
        ("rotor_models", "grid9", 3, True),
        ("rotor_models", "grid9_raw", 3, False),
        ("rotor_models", "level4", 4, True),
        ("rotor_models", "level4_raw", 4, False),
    ],
)
def test_quadrature_factory_tokens(collection, name, count, reduce):
    model = ModelBook().sources[collection][name]
    assert model.n == count
    assert model.reduce is reduce


def test_partial_grid_factory_converts_total_points_to_rotor_side():
    assert ModelBook().partial_wakes["grid9"].rotor.n == 3


@pytest.mark.parametrize(
    "name, minutes",
    [("timelines_30s", 0.5), ("timelines_2min", 2.0), ("seq_dyn_wakes_60s", 1.0)],
)
def test_wake_frame_duration_tokens(name, minutes):
    assert ModelBook().wake_frames[name].dt_min == minutes


def test_wake_frame_duration_rejects_unsupported_units():
    with pytest.raises(NotImplementedError, match="into minutes"):
        ModelBook().wake_frames["timelines_5h"]


def test_model_lookup_creates_and_reuses_one_named_instance():
    book = ModelBook()
    model = book.get("outputs", "custom", "FarmResultsOutput", variables=["P"])
    assert model is book.get("outputs", "custom")
    assert model.fdata_variables == ["P"]


def test_model_lookup_reports_unknown_collection_and_model():
    book = ModelBook()
    with pytest.raises(KeyError, match="unknown"):
        book["unknown"]
    with pytest.raises(KeyError, match="not found"):
        book.get("outputs", "unknown")


def test_model_book_contents_filter_models_and_factories(capsys):
    ModelBook().print_toc(subset=["rotor_models"], search="grid")
    output = capsys.readouterr().out
    assert output.startswith("rotor_models\n")
    assert "grid<n2>" in output
    assert "level<n>" not in output
    assert "wake_models\n" not in output


def test_noninteractive_test_policy_survives_local_monkeypatch_cleanup(monkeypatch):
    monkeypatch.undo()
    book = ModelBook()
    assert all(output.show is False for output in book.outputs.values())
    assert book.outputs["flow_plot_xy_mean"].show is False


@pytest.mark.parametrize(
    "name,group",
    [
        ("farms_results", "wind_farm"),
        ("cluster_results", "cluster"),
        ("total_results", "all"),
    ],
)
def test_plain_grouped_presets_are_explicit_and_listed(name, group, capsys):
    book = ModelBook()
    assert name in book.outputs.keys()
    output = book.outputs[name]
    assert type(output) is FarmResultsOutput
    assert output.aggregate_turbines == group
    assert output.aggregate_states is False
    assert output.ambient_vars and output.waked_vars
    assert FV.P in output.fdata_variables
    book.print_toc(subset=["outputs"], search=name)
    assert f"{name}: " in capsys.readouterr().out


@pytest.mark.parametrize(
    "collection",
    [
        "rotor_models",
        "turbine_models",
        "partial_wakes",
        "wake_frames",
        "wake_deflections",
        "wake_superpositions",
        "wake_models",
        "ground_models",
        "outputs",
    ],
)
def test_factory_templates_are_unique(collection):
    factories = ModelBook().sources[collection].factories
    templates = [factory.name_template for factory in factories]
    assert len(templates) == len(set(templates))


def _assert_factory_reachable(registry, intended, name):
    """Compare actual lookup with the first constructible factory's full parameters."""
    expected = intended.construct(name)
    actual = registry[name]
    assert type(actual) is type(expected), (name, intended.name_template)
    _, expected_kwargs = intended.check_match(name, error=True, ret_pars=True)
    for candidate in registry.factories:
        try:
            candidate.construct(name)
        except ValueError:
            continue
        _, actual_kwargs = candidate.check_match(name, error=True, ret_pars=True)
        assert candidate.base is intended.base, (name, candidate.name_template)
        assert candidate.args == intended.args, (name, candidate.name_template)
        assert actual_kwargs == expected_kwargs, (name, candidate.name_template)
        return
    pytest.fail(f"No factory constructed its valid example {name!r}")


@pytest.mark.parametrize(
    "collection",
    [
        "rotor_models",
        "turbine_models",
        "partial_wakes",
        "wake_frames",
        "wake_deflections",
        "wake_superpositions",
        "wake_models",
        "ground_models",
    ],
)
def test_factory_order_preserves_every_closed_option_and_numeric_variants(collection):
    registry = ModelBook().sources[collection]
    registry.store_created = False
    for factory in registry.factories:
        tokens = {
            variable: list(factory.options[variable])
            if isinstance(factory.options.get(variable), dict)
            else _FACTORY_TOKENS[variable]
            for variable in factory.variables
        }
        for name in factory.get_examples(**tokens):
            _assert_factory_reachable(registry, factory, name)
        if factory.example is not None:
            _assert_factory_reachable(registry, factory, factory.example)


@pytest.mark.parametrize(
    "token,expected",
    [
        ("004", 0.04),
        ("0.04", 0.04),
        ("0001", 0.001),
        ("0.001", 0.001),
        ("0", 0.0),
        ("1", 1.0),
        ("1.5", 1.5),
        ("4e-2", 0.04),
    ],
)
def test_coefficient_tokens_preserve_compact_and_standard_numbers(token, expected):
    assert _parse_coefficient(token) == expected


@pytest.mark.parametrize("token", ["", "invalid", "0..04"])
def test_coefficient_tokens_reject_invalid_numbers(token):
    with pytest.raises(ValueError):
        _parse_coefficient(token)


@pytest.mark.parametrize("suffix", ["", "_amb", "_target", "_amb_target", "_loc"])
def test_vector_superpositions_exclude_unsupported_limits(suffix):
    registry = ModelBook().wake_superpositions
    supported = registry[f"vector{suffix}"]
    assert supported.scale_amb is (suffix in ["_amb", "_amb_target", "_loc"])
    assert supported.scale_target is (suffix in ["_target", "_amb_target", "_loc"])
    with pytest.raises(KeyError):
        registry[f"vector{suffix}_lim"]


def test_integrated_turbopark_factories_only_advertise_supported_growth_modes():
    registry = ModelBook().wake_models
    templates = [
        factory.name_template
        for factory in registry.factories
        if factory.name_template.startswith("TurbOParkIX_")
    ]
    assert templates == [
        "TurbOParkIX_<superposition>_k<k>_dx<dx>",
        "TurbOParkIX_<superposition>_dx<dx>",
    ]
    assert registry["TurbOParkIX_linear_k004_dx10"].wake_k.is_kTI is False
    assert registry["TurbOParkIX_linear_dx10"].wake_k.all_none


@pytest.mark.parametrize(
    "prefix", ["slice_data_", "slices_data_", "flow_plot_", "chunk_gif_", "chunk_mp4_"]
)
def test_non_result_output_factories_preserve_every_option(prefix):
    registry = ModelBook().outputs
    registry.store_created = False
    factories = [
        factory
        for factory in registry.factories
        if factory.name_template.startswith(prefix)
    ]
    assert len(factories) == 1
    factory = factories[0]
    tokens = {
        variable: list(factory.options[variable])
        if isinstance(factory.options.get(variable), dict)
        else _FACTORY_TOKENS[variable]
        for variable in factory.variables
    }
    for name in factory.get_examples(**tokens):
        _assert_factory_reachable(registry, factory, name)
