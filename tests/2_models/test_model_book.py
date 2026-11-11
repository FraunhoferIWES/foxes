"""Registry ownership, factory conversion and lookup after setup decomposition."""

import pytest

from foxes.models import ModelBook


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
