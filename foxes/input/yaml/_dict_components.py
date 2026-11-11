"""Construct configured YAML components without retaining execution state."""

from pathlib import Path
from typing import Any

import foxes.constants as FC
import foxes.input.farm_layout as farm_layout
from foxes.config import config
from foxes.core import Engine, WindFarm
from foxes.models import ModelBook
from foxes.utils import Dict


def _print(verbosity: int | None, *args: Any, level: int = 1, **kwargs: Any) -> None:
    if verbosity is None or verbosity >= level:
        print(*args, **kwargs)


def _configure_paths(
    work_dir: Path | str | None,
    input_dir: Path | str | None,
    output_dir: Path | str | None,
    verbosity: int | None,
) -> None:
    """Apply explicit path overrides and report the effective directories."""
    for key, path in zip(
        (FC.WORK_DIR, FC.INPUT_DIR, FC.OUTPUT_DIR), (work_dir, input_dir, output_dir)
    ):
        if path is not None:
            config[key] = path
    _print(
        verbosity,
        "\n--------------------- Reading foxes parameter dict ---------------------",
    )
    _print(verbosity, "Working directory  :", config.work_dir)
    _print(verbosity, "Input directory    :", config.input_dir)
    _print(verbosity, "Output directory   :", config.output_dir)


def _read_model_book(idict: Dict[Any, Any], verbosity: int | None) -> ModelBook:
    """Create independent model registries and consume configured model names."""
    book = ModelBook()
    if "model_book" in idict:
        _print(verbosity, "Creating model book")
        definitions = idict.get_item("model_book")
        for source, entries in definitions.items():
            registry = book.sources.get_item(source)
            model_type = book.base_classes.get_item(source)
            for index, entry in enumerate(entries):
                parameters: Dict[str, Any] = Dict(
                    entry, _name=f"{definitions.name}.s.{index}"
                )
                name = parameters.pop_item("name")
                _print(verbosity, f"  Adding {source}.{name}")
                registry[name] = model_type.new(**parameters)
    return book


def _read_farm(idict: Dict[Any, Any], verbosity: int | None) -> WindFarm:
    """Create a farm and consume its configured layout definitions."""
    _print(verbosity, "Creating wind farm")
    parameters = idict.get_item("wind_farm")
    layouts = parameters.pop_item("layouts")
    farm = WindFarm(**parameters)
    for layout in layouts:
        add_layout = getattr(farm_layout, layout.pop_item("function"))
        if verbosity is not None:
            layout["verbosity"] = max(verbosity - 1, 0)
        add_layout(farm, **layout)
    return farm


def _read_engine(options: dict[str, Any] | None, verbosity: int | None) -> Engine:
    """Create an engine with explicit options overriding inherited verbosity."""
    if options is None:
        _print(verbosity, "Using default engine")
        return Engine.new(
            engine_type="default",
            verbosity=1 if verbosity is None else max(verbosity - 1, 0),
        )
    parameters = dict(options)
    if verbosity is not None:
        parameters.setdefault("verbosity", max(verbosity - 1, 0))
    engine = Engine.new(**parameters)
    _print(verbosity, f"Using engine: {engine}")
    return engine
