import pandas as pd
from xarray import Dataset
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.artist import Artist

import foxes.input.farm_layout as farm_layout
from foxes.core import States, Engine, WindFarm, Algorithm
from foxes.models import ModelBook
from foxes.core import Output, OutputList
from foxes.models.outputs import FarmResultsOutput
from foxes.utils import Dict
from foxes.config import config, get_output_path
import foxes.constants as FC


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
    """
    Read dictionary input into foxes objects

    Parameters
    ----------
    idict
        The input parameter dictionary
    farm
        The wind farm, overrules settings from idict
    states
        The ambient states, overrules settings from idict
    mbook
        The model book, overrules settings from idict
    algo
        The algorithm, overrules settings from idict
    engine_pars
        Parameters for engine creation, overrules
        settings from idict
    iterative
        Force iterative calculations, overrules
        settings from idict
    verbosity
        Force a verbosity level, 0 = silent, overrules
        settings from idict
    work_dir
        Path to the working directory
    input_dir
        The default input directory
    output_dir
        The default output directory
    algo_pars
        Additional parameters for the algorithm, overrules
        settings from idict

    Returns
    -------
    algo
        The algorithm
    engine
        The engine, or None if not set


    """

    def _print(*args: Any, level: int = 1, **kwargs: Any) -> None:
        if verbosity is None or verbosity >= level:
            print(*args, **kwargs)

    # set working directory:
    ld = 0
    for c, d in zip(
        [FC.WORK_DIR, FC.INPUT_DIR, FC.OUTPUT_DIR], [work_dir, input_dir, output_dir]
    ):
        if d is not None:
            config[c] = d
            ld = max(ld, len(str(d)))
    _print("\n--------------------- Reading foxes parameter dict ---------------------")
    _print("Working directory  :", config.work_dir)
    _print("Input directory    :", config.input_dir)
    _print("Output directory   :", config.output_dir)

    # create states:
    if states is None:
        if algo is None:
            _print("Creating states")
            states = States.new(**idict["states"])
        else:
            states = algo.states
    else:
        assert algo is None, (
            "Cannot handle both the algo and the states argument, please drop one"
        )

    # create model book:
    if mbook is None:
        if algo is None:
            mbook = ModelBook()
            if "model_book" in idict:
                _print("Creating model book")
                mdict = idict.get_item("model_book")
                for s, mlst in mdict.items():
                    t = mbook.sources.get_item(s)
                    c = mbook.base_classes.get_item(s)
                    ms: list[Dict[Any, Any]] = [
                        Dict(m, _name=f"{mdict.name}.s.{i}") for i, m in enumerate(mlst)
                    ]
                    for m in ms:
                        mname = m.pop_item("name")
                        _print(f"  Adding {s}.{mname}")
                        t[mname] = c.new(**m)
        else:
            mbook = algo.mbook
    else:
        assert algo is None, (
            "Cannot handle both the algo and the mbook argument, please drop one"
        )

    # create farm:
    if farm is None:
        if algo is None:
            _print("Creating wind farm")
            fdict = idict.get_item("wind_farm")
            lyts = fdict.pop_item("layouts")
            farm = WindFarm(**fdict)
            for lyt in lyts:
                add_fun = getattr(farm_layout, lyt.pop_item("function"))
                if verbosity is not None:
                    lyt["verbosity"] = max(verbosity - 1, 0)
                add_fun(farm, **lyt)
        else:
            farm = algo.farm
    else:
        assert algo is None, (
            "Cannot handle both the algo and the farm argument, please drop one"
        )

    # create engine:
    engine = None
    if engine_pars is not None:
        options = dict(engine_pars)
        if verbosity is not None:
            options.setdefault("verbosity", max(verbosity - 1, 0))
        engine = Engine.new(**options)
        _print(f"Using engine: {engine}")
    elif "engine" in idict:
        options = dict(idict["engine"])
        if verbosity is not None:
            options.setdefault("verbosity", max(verbosity - 1, 0))
        engine = Engine.new(**options)
        _print(f"Using engine: {engine}")
    else:
        _print("Using default engine")
        engine = Engine.new(
            engine_type="default",
            verbosity=1 if verbosity is None else max(verbosity - 1, 0),
        )

    # create algorithm:
    if algo is None:
        adict = idict.get_item("algorithm")
        if iterative is not None and iterative:
            adict["algo_type"] = "Iterative"
        _print("Creating algorithm :", adict["algo_type"])
        adict.update(dict(farm=farm, states=states, mbook=mbook))
        if verbosity is not None:
            adict["verbosity"] = max(verbosity - 1, 0)
        if algo_pars is not None:
            adict.update({v: d for v, d in algo_pars.items() if d is not None})
        algo = Algorithm.new(**adict)

    _print("------------------------------------------------------------------------\n")

    return algo, engine


def _get_object(results_storage: dict[Any, Any], d: str) -> Any:
    """Helper function for object extraction"""
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
    for key, value in definition.items():
        definition[key] = _resolve_labels(value, results_storage)
    output = Output.new(output_type=output_type, **definition)
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
        outputs=output, farm_results=farm_results, point_results=point_results
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
        Constructor-defined outputs, or ``output_type: plt``/Matplotlib
        ``object: $name`` entries with a ``functions`` list.
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
        invalid constructor options, plotting functions or result labels.
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

    def _print(*args: Any, level: int = 1, **kwargs: Any) -> None:
        if verbosity is None or verbosity >= level:
            print(*args, **kwargs)

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
        # run farm calculation:
        rdict = idict.get_item("calc_farm", Dict(_name=idict.name + ".calc_farm"))
        if rdict.pop_item("run", True):
            _print("Running calc_farm")
            farm_results = algo.calc_farm(**rdict)
        else:
            farm_results = None
        out: tuple[Any, ...] = (farm_results,)

        # Report collected farm data independently of later output requirements.
        if farm_results is not None and (verbosity is None or verbosity > 1):
            collected = farm_results
            if isinstance(collected, dict):
                collected = collected.get(FC.FARM_RESULTS)
            else:
                selectors = rdict.get("outputs", "farm_results")
                pending = (
                    list(selectors) if isinstance(selectors, list) else [selectors]
                )
                while pending:
                    selected = pending.pop()
                    if isinstance(selected, str):
                        selected = algo.mbook.outputs[selected]
                    if isinstance(selected, FarmResultsOutput):
                        break
                    if isinstance(selected, OutputList):
                        pending.extend(selected.outputs)
                else:
                    collected = None
            if collected is not None:
                _print("\nFarm results:\n", collected, level=2)

        # run points calculation:
        point_results = None
        if "calc_points" in idict:
            rdict = idict.get_item("calc_points")
            if rdict.pop_item("run"):
                _print("Running calc_points")
                points = rdict.pop_item("points")
                if isinstance(points, str):
                    _print("Reading file", points)
                    points = pd.read_csv(points).to_numpy()
                point_results = algo.calc_points(farm_results, points=points, **rdict)
            else:
                point_results = None
            out += (point_results,)

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
