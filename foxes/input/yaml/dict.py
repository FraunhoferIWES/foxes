"""Read and execute constructor-defined FOXES YAML parameter dictionaries."""

import numpy as np
import pandas as pd
from xarray import Dataset
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.artist import Artist

from foxes.core import States, Engine, WindFarm, Algorithm
from foxes.models import ModelBook
from foxes.core import Output, OutputList
from foxes.models.outputs import FarmResultsOutput
from foxes.models.outputs._point_data_output import PointDataOutput
from foxes.utils import Dict
from foxes.config import get_output_path
import foxes.constants as FC

from ._dict_components import (
    _configure_paths,
    _print,
    _read_engine,
    _read_farm,
    _read_model_book,
)


def read_dict(
    idict: Dict[Any, Any],
    farm: WindFarm | None = None,
    states: States | None = None,
    mbook: ModelBook | None = None,
    algo: Algorithm | None = None,
    engine_pars: dict[str, Any] | None = None,
    iterative: bool | None = None,
    verbosity: int | None = None,
    work_dir: Path | str | None = None,
    input_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    **algo_pars: Any,
) -> tuple[Algorithm, Engine | None]:
    """Read dictionary input into FOXES objects.

    Parameters
    ----------
    idict
        Input parameter dictionary, consumed when constructing components.
    farm
        Wind farm overriding settings from ``idict``.
    states
        Ambient states overriding settings from ``idict``.
    mbook
        Model book overriding settings from ``idict``.
    algo
        Existing algorithm overriding settings from ``idict``. Do not combine
        this with explicit ``farm``, ``states`` or ``mbook`` arguments.
    engine_pars
        Engine constructor parameters overriding settings from ``idict``.
    iterative
        Force Iterative calculations, overriding settings from ``idict``.
    verbosity
        Force a verbosity level, where zero is silent, overriding settings
        from ``idict``.
    work_dir
        Path to the working directory.
    input_dir
        Default input directory.
    output_dir
        Default output directory.
    algo_pars
        Additional algorithm parameters overriding settings from ``idict``.

    Returns
    -------
    algo
        Existing or newly constructed algorithm.
    engine
        The configured engine, or the default engine when none is specified.
    """

    _configure_paths(work_dir, input_dir, output_dir, verbosity)

    # Explicit components must not compete with an existing algorithm's ownership.
    if algo is not None:
        for label, component in (("states", states), ("mbook", mbook), ("farm", farm)):
            assert component is None, (
                f"Cannot handle both the algo and the {label} argument, please drop one"
            )
        states, mbook, farm = algo.states, algo.mbook, algo.farm
    else:
        if states is None:
            _print(verbosity, "Creating states")
            states = States.new(**idict["states"])
        if mbook is None:
            mbook = _read_model_book(idict, verbosity)
        if farm is None:
            farm = _read_farm(idict, verbosity)

    # create engine:
    engine = _read_engine(
        engine_pars if engine_pars is not None else idict.get("engine"), verbosity
    )

    # create algorithm:
    if algo is None:
        adict = idict.get_item("algorithm")
        if iterative is not None and iterative:
            adict["algo_type"] = "Iterative"
        _print(verbosity, "Creating algorithm :", adict["algo_type"])
        adict.update(dict(farm=farm, states=states, mbook=mbook))
        if verbosity is not None:
            adict["verbosity"] = max(verbosity - 1, 0)
        adict.update({v: d for v, d in algo_pars.items() if d is not None})
        algo = Algorithm.new(**adict)

    _print(
        verbosity,
        "------------------------------------------------------------------------\n",
    )

    return algo, engine


def _get_object(results_storage: dict[Any, Any], d: str) -> Any:
    """Extract a stored result with optional tuple-of-integer indexing."""
    d = d.replace("]", "")
    i0 = d.find("[")
    if i0 > 0:
        inds = tuple([int(x) for x in d[i0 + 1 :].split(",")])
        return results_storage[d[:i0]][inds]
    else:
        return results_storage[d]


def _resolve_labels(value: Any, storage: dict[Any, Any]) -> Any:
    """Resolve stored-result references in nested constructor configuration.

    Parameters
    ----------
    value
        Configuration value, including nested mappings, lists and tuples.
        Strings beginning with ``$`` name stored results and may include the
        integer indexing syntax supported by ``_get_object``.
    storage
        Reference-name-to-result mapping, read without modification.

    Returns
    -------
    resolved_value
        Rebuilt containers with references replaced by the stored objects.
        Referenced objects and non-container values are not copied.

    Raises
    ------
    KeyError
        If a referenced result is unavailable.
    ValueError
        If a reference's positional indices cannot be parsed as integers.
    """
    # Rebuild configuration containers while preserving referenced object identity.
    if isinstance(value, str) and value.startswith("$"):
        return _get_object(storage, value)
    if isinstance(value, dict):
        return {key: _resolve_labels(item, storage) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_resolve_labels(item, storage) for item in value)
    return value


def _store_labels(
    storage: dict[Any, Any], labels: str | list[str] | tuple[str, ...], result: Any
) -> None:
    """Store named result references for later output entries.

    Parameters
    ----------
    storage
        Result-reference mapping, updated in place without copying result values.
    labels
        A single ``$name`` for the entire result, or a sequence paired with
        successive result elements. Empty labels and ``none``, ``None``, ``_``
        and ``__`` are skipped. Sequence pairing stops at the shorter input.
    result
        Returned output value or iterable of values to register.

    Raises
    ------
    ValueError
        If a non-skipped label lacks ``$`` or contains brackets or a comma.
        Earlier valid entries remain stored if a later label is invalid.
    """
    # Share result objects rather than duplicating scientific arrays.
    entries = [(labels, result)] if isinstance(labels, str) else zip(labels, result)
    for label, item in entries:
        if label in ("", "none", "None", "_", "__"):
            continue
        if not label.startswith("$") or any(char in label for char in "[],"):
            raise ValueError(
                f"Result labels must be simple '$name' references: {label!r}"
            )
        storage[label] = item


def _output_states(
    index: int,
    definition: Dict[Any, Any],
    algo: Algorithm,
    results_storage: dict[Any, Any],
) -> list[Any] | None:
    """Consume state references and resolve positions before output reduction."""
    positions = _resolve_labels(definition.pop("states_isel", None), results_storage)
    labels = _resolve_labels(definition.pop("states_sel", None), results_storage)
    if positions is not None and labels is not None:
        raise ValueError(
            f"Output {index}: Select states by positions or labels, not both"
        )
    if positions is None:
        return labels
    if not algo.initialized:
        algo.initialize()
    indices = np.asarray(positions)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or not len(indices)
        or any(isinstance(i, (bool, np.bool_)) for i in positions)
    ):
        raise ValueError(
            f"Output {index}: states_isel requires nonempty integer positions"
        )
    states = np.asarray(algo.states.index())
    if np.any(indices < -len(states)) or np.any(indices >= len(states)):
        raise ValueError(f"Output {index}: State position outside the input states")
    return list(states[indices])


def _run_output_model(
    index: int,
    definition: Dict[Any, Any],
    algo: Algorithm | None,
    farm_results: Dataset | None,
    point_results: Any,
    results_storage: dict[Any, Any],
    nofig: bool,
) -> Any:
    """Construct and execute one constructor-defined output in its engine phase.

    Parameters
    ----------
    index
        Output-entry position used in contextual diagnostics.
    definition
        Mutable constructor configuration. ``output_type`` and ``result_labels``
        are consumed, and nested result references are resolved in place.
    algo
        Initialized algorithm exposing ``calc_outputs``.
    farm_results
        Supplied farm fields on ``(FC.STATE, FC.TURBINE)``, forwarded without
        mutation or additional evaluation columns.
    point_results
        Supplied point fields, forwarded to ``calc_outputs`` without copying.
    results_storage
        Shared result-reference mapping, read during construction and updated
        with explicitly requested result labels.
    nofig
        Suppress figure creation even if the output configuration enables it.

    Returns
    -------
    result
        The value returned by ``algo.calc_outputs``.

    Raises
    ------
    ValueError
        If the algorithm is missing, construction produces no model, the model
        is outside ``foxes.models.outputs``, or a result label is invalid.
    TypeError
        If the algorithm does not provide ``calc_outputs``.
    KeyError
        If a referenced result or required constructor option is unavailable.
    """
    # Validate execution support before consuming constructor options.
    if algo is None:
        raise ValueError(f"Output {index}: Algorithm is required")
    if not hasattr(algo, "calc_outputs"):
        raise TypeError(f"Output {index}: Algorithm must implement calc_outputs")

    # Resolve nested references before constructing the output model.
    output_type = definition.pop_item("output_type")
    labels = definition.pop("result_labels", None)
    state_labels = _output_states(index, definition, algo, results_storage)
    for key, value in definition.items():
        definition[key] = _resolve_labels(value, results_storage)
    if hasattr(algo, "mbook") and output_type in algo.mbook.outputs:
        if definition:
            raise ValueError(
                f"Output {index}: Named preset '{output_type}' does not accept "
                "constructor overrides; configure model_book.outputs or use the class name"
            )
        output = deepcopy(algo.mbook.outputs[output_type])
    else:
        output = Output.new(output_type=output_type, **definition)
    if state_labels is not None and isinstance(output, PointDataOutput):
        # Grid consumers also use this selection to restore requested state order.
        output.states_sel = state_labels
        output.states_isel = None
    if not type(output).__module__.startswith("foxes.models.outputs."):
        raise ValueError(
            f"Output {index}: Constructor-defined outputs must belong "
            "to foxes.models.outputs"
        )

    # Execute through the ordinary output lifecycle and publish pipeline results.
    output.nofig = nofig or output.nofig
    if output.nofig:
        output.show = False
    result = algo.calc_outputs(
        outputs=output,
        farm_results=farm_results,
        point_results=point_results,
        **({"states_sel": state_labels} if state_labels is not None else {}),
    )
    if labels is not None:
        _store_labels(results_storage, labels, result)
    return result


def _run_plot_functions(
    definition: Dict[Any, Any],
    storage: dict[Any, Any],
    nofig: bool,
) -> list[Any]:
    """Run YAML-only pyplot or stored Matplotlib artist methods.

    No output model, algorithm injection or evaluation-method dispatch is used.
    Figure-less runs skip every call and register ``None`` for result labels,
    so later constructor references remain resolvable without creating figures.
    """
    reference = definition.pop("object", None)
    definition.pop("output_type", None)
    functions = definition.pop_item("functions")
    if definition:
        raise ValueError(f"Unexpected YAML plotting options: {list(definition)}")
    obj = (
        None if nofig else plt if reference is None else _get_object(storage, reference)
    )
    if not nofig and obj is not plt and not isinstance(obj, Artist):
        raise TypeError("YAML object plotting requires a stored Matplotlib artist")
    results = []
    for parameters in functions:
        options: Dict[Any, Any] = Dict(parameters)
        name = options.pop_item("function")
        labels = options.pop("result_labels", None)
        result = None
        if not nofig:
            method = getattr(obj, name, None) if not name.startswith("_") else None
            if not callable(method):
                raise ValueError(f"Unknown public Matplotlib function: {name!r}")
            options = _resolve_labels(options, storage)
            args = options.pop("args", ())
            if name == "savefig":
                if args:
                    args = (get_output_path(args[0]), *args[1:])
                elif "fname" in options:
                    options["fname"] = get_output_path(options["fname"])
            result = method(*args, **options)
        if labels is not None:
            value = (
                [None] * len(labels)
                if nofig and not isinstance(labels, str)
                else result
            )
            _store_labels(storage, labels, value)
        results.append(result)
    return results


def run_outputs(
    idict: Dict[Any, Any],
    algo: Algorithm | None = None,
    farm_results: Dataset | None = None,
    point_results: Any = None,
    with_engine: bool = False,
    results_storage: Dict[Any, Any] | None = None,
    ret_results_storage: bool = False,
    nofig: bool = False,
    verbosity: int | None = None,
) -> tuple[list[tuple[dict[str, Any], list[Any] | None]], Dict[Any, Any]]:
    """Run output models and the YAML-only Matplotlib exception for one phase.

    Parameters
    ----------
    idict
        Outputs with a class name or model-book preset in ``output_type``.
        Presets are copied and accept no constructor overrides; customize the
        model book or use a class name. ``states_sel`` selects state labels and
        ``states_isel`` selects integer positions before reductions. They are
        mutually exclusive. ``output_type: plt``/Matplotlib ``object: $name``
        entries instead have a ``functions`` list.
    algo
        Algorithm exposing the ordinary ``calc_outputs`` lifecycle.
    farm_results
        Supplied farm fields, never captured by an output constructor.
    point_results
        Supplied point fields, forwarded without scientific recalculation.
    with_engine
        Execute entries matching this phase. Models default to ``True``;
        YAML-only Matplotlib calls default to ``False``.
    results_storage
        Shared result references resolved recursively in constructor options.
    ret_results_storage
        Request reference storage explicitly; both settings return the same pair.
    nofig
        Suppress model figures and skip all YAML-only Matplotlib calls.
    verbosity
        Progress verbosity; zero is silent.

    Returns
    -------
    outputs
        Ordered definition/result pairs. Skipped phase entries have ``None``;
        model entries have a single-element result list, and Matplotlib entries
        have one value per function call.
    results_storage
        Supplied or newly created result-reference mapping.

    Raises
    ------
    ValueError
        If a model entry uses imperative methods, lacks an algorithm, or has
        invalid constructor options, state selections, preset overrides,
        plotting functions or result labels.
    KeyError
        If an output type or referenced result is unavailable.
    TypeError
        If ``object`` refers to something other than a Matplotlib artist.
    """
    if results_storage is None:
        results_storage = Dict(_name="result_storage")
    outputs: list[tuple[dict[str, Any], list[Any] | None]] = []
    for index, entry in enumerate(idict.get("outputs", [])):
        definition = deepcopy(entry)
        plotting = definition.get("output_type") == "plt" or "object" in definition
        if "functions" in definition and not plotting:
            raise ValueError(
                f"Output {index}: Configure an output_type through constructor options"
            )
        selected_engine = definition.pop("with_engine", not plotting)
        if selected_engine != with_engine:
            outputs.append((dict(entry), None))
            continue
        if verbosity is None or verbosity > 0:
            print(f"Running output {index}: {definition.get('output_type')}")
        if plotting:
            results = _run_plot_functions(definition, results_storage, nofig)
            outputs.append((dict(entry), results))
            continue
        result = _run_output_model(
            index,
            definition,
            algo,
            farm_results,
            point_results,
            results_storage,
            nofig,
        )
        outputs.append((dict(entry), [result]))
    return outputs, results_storage


def _collected_farm_results(algo: Algorithm, result: Any, selectors: Any) -> Any:
    """Find a collected farm dataset without treating other outputs as farm data."""
    if isinstance(result, dict):
        return result.get(FC.FARM_RESULTS)
    pending = list(selectors) if isinstance(selectors, list) else [selectors]
    while pending:
        selected = pending.pop()
        if isinstance(selected, str):
            selected = algo.mbook.outputs[selected]
        if isinstance(selected, FarmResultsOutput):
            return result
        if isinstance(selected, OutputList):
            pending.extend(selected.outputs)
    return None


def _run_calculations(
    idict: Dict[Any, Any],
    algo: Algorithm,
    verbosity: int | None,
) -> tuple[Any, Any, tuple[Any, ...]]:
    """Consume calculation switches and return raw farm/point results in order."""
    parameters = idict.get_item("calc_farm", Dict(_name=idict.name + ".calc_farm"))
    farm_results = None
    if parameters.pop_item("run", True):
        _print(verbosity, "Running calc_farm")
        farm_results = algo.calc_farm(**parameters)
    out: tuple[Any, ...] = (farm_results,)
    if farm_results is not None and (verbosity is None or verbosity > 1):
        collected = _collected_farm_results(
            algo, farm_results, parameters.get("outputs", "farm_results")
        )
        if collected is not None:
            _print(verbosity, "\nFarm results:\n", collected, level=2)
    point_results = None
    if "calc_points" in idict:
        parameters = idict.get_item("calc_points")
        if parameters.pop_item("run"):
            _print(verbosity, "Running calc_points")
            points = parameters.pop_item("points")
            if isinstance(points, str):
                _print(verbosity, "Reading file", points)
                points = pd.read_csv(points).to_numpy()
            point_results = algo.calc_points(
                farm_results=farm_results, points=points, **parameters
            )
        out += (point_results,)
    return farm_results, point_results, out


def run_dict(
    idict: Dict[Any, Any],
    farm: WindFarm | None = None,
    states: States | None = None,
    mbook: ModelBook | None = None,
    algo: Algorithm | None = None,
    engine_pars: dict[str, Any] | None = None,
    iterative: bool | None = None,
    nofig: bool = False,
    verbosity: int | None = None,
    work_dir: Path | str | None = None,
    input_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    **algo_pars: Any,
) -> tuple[Any, ...]:
    """Run configured farm/point calculations and output consumers.

    Parameters
    ----------
    idict
        FOXES parameter dictionary. Nested calculation sections are consumed
        in place when reading their execution switches.
    farm
        Wind farm overriding the corresponding dictionary configuration.
    states
        Ambient states overriding the dictionary configuration.
    mbook
        Model book overriding the dictionary configuration.
    algo
        Algorithm overriding the dictionary configuration.
    engine_pars
        Engine constructor parameters overriding the dictionary configuration.
    iterative
        Select an Iterative algorithm, overriding the dictionary configuration.
    nofig
        Suppress figures, overriding configured output display settings.
    verbosity
        Force a verbosity level, overriding settings from ``idict``.
        Zero is silent; levels two and higher print collected farm results.
        ``None`` retains unrestricted diagnostics.
    work_dir
        Working directory overriding the dictionary configuration.
    input_dir
        Base directory for relative input paths.
    output_dir
        Base directory for relative output paths.
    algo_pars
        Additional algorithm configuration passed to ``read_dict()``.

    Returns
    -------
    execution_results
        Tuple starting with the farm calculation result, or ``None`` when
        disabled. A point result is appended when ``calc_points`` is configured,
        followed by the configured output-consumer entries. Farm calculation
        results retain the dataset or named-result mapping returned by the
        algorithm; reporting does not unwrap or mutate the returned result.

    Notes
    -----
    Each collected farm-results dataset is printed once at verbosity two or
    higher, or with unrestricted verbosity. This does not require the fields
    used by later reports. Writer-only and non-farm results are not printed as
    farm datasets. Scientific component verbosity is reduced inside the reader.
    """

    # read components:
    algo_pars.pop("verbosity", None)
    algo, engine = read_dict(
        idict,
        farm=farm,
        states=states,
        mbook=mbook,
        algo=algo,
        engine_pars=engine_pars,
        iterative=iterative,
        verbosity=verbosity,
        work_dir=work_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        **algo_pars,
    )
    results_storage = None
    assert engine is not None
    with engine:
        farm_results, point_results, out = _run_calculations(idict, algo, verbosity)

        # run outputs with engine:
        out_w, results_storage = run_outputs(
            idict,
            algo,
            farm_results,
            point_results,
            with_engine=True,
            nofig=nofig,
            results_storage=results_storage,
            ret_results_storage=True,
            verbosity=verbosity,
        )
        out_w = list(out_w)

    # run outputs w/o engine:
    out_wo = list(
        run_outputs(
            idict,
            algo,
            farm_results,
            point_results,
            with_engine=False,
            nofig=nofig,
            results_storage=results_storage,
            verbosity=verbosity,
        ),
    )

    # combine outputs:
    out += tuple(a if a is not None else b for a, b in zip(out_w, out_wo))

    return out
