import matplotlib.pyplot as plt
import pandas as pd
from xarray import Dataset
from inspect import signature
from copy import deepcopy
from pathlib import Path
from typing import Any

import foxes.input.farm_layout as farm_layout
from foxes.core import States, Engine, WindFarm, Algorithm
from foxes.models import ModelBook
from foxes.core import Output
from foxes.utils import Dict, new_cls
from foxes.config import config
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
                    lyt["verbosity"] = verbosity - 1
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
        engine = Engine.new(**engine_pars)
        _print(f"Using engine: {engine}")
    elif "engine" in idict:
        if verbosity is not None:
            idict["verbosity"] = verbosity - 1
        engine = Engine.new(**idict["engine"])
        _print(f"Using engine: {engine}")
    else:
        _print("Using default engine")
        engine = Engine.new(engine_type="default")

    # create algorithm:
    if algo is None:
        adict = idict.get_item("algorithm")
        if iterative is not None and iterative:
            adict["algo_type"] = "Iterative"
        _print("Creating algorithm :", adict["algo_type"])
        adict.update(dict(farm=farm, states=states, mbook=mbook))
        if verbosity is not None:
            adict["verbosity"] = verbosity - 1
        if algo_pars is not None:
            adict.update({v: d for v, d in algo_pars.items() if d is not None})
        algo = Algorithm.new(**adict)

    _print("------------------------------------------------------------------------\n")

    return algo, engine


def get_output_obj(
    ocls: str,
    odict: dict[str, Any],
    algo: Algorithm | None,
    farm_results: Dataset | None = None,
    point_results: Any = None,
    base_class: type[Output] = Output,
    extra_sig: dict[str, Any] = {},
) -> Output | None:
    """
    Create the output object

    Parameters
    ----------
    ocls
        Name of the output class
    odict
        The output dict
    algo
        The algorithm
    farm_results
        The farm results
    point_results
        The point results
    base_class
        The output's base class
    extra_sig
        Extra function signature check, sets
        arguments (key) with data (value)

    Returns
    -------
    obj
        The output object


    """
    cls = new_cls(base_class, ocls)
    assert cls is not None, f"Output class '{ocls}' was not found"
    prs = list(signature(cls.__init__).parameters.keys())
    if "algo" in prs:
        assert algo is not None, f"Output of type '{ocls}' requires algo"
        odict["algo"] = algo
    if "farm" in prs:
        assert algo is not None, f"Output of type '{ocls}' requires algo"
        odict["farm"] = algo.farm
    if "farm_results" in prs:
        if farm_results is None:
            print(f"No farm results; skipping output {ocls}")
            return None
        odict["farm_results"] = farm_results
    if "point_results" in prs:
        odict["point_results"] = point_results
    for k, v in extra_sig.items():
        if k in prs:
            odict[k] = v

    return cls(**odict)


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
    extra_sig: dict[str, Any],
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
        Supplied farm fields on ``(FC.STATE, FC.TURBINE)``. An ``add_*`` output
        updates this dataset in place for subsequent pipeline entries.
    point_results
        Supplied point fields, forwarded to ``calc_outputs`` without copying.
    extra_sig
        Additional constructor arguments injected by matching parameter names.
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
    output = get_output_obj(
        output_type,
        definition,
        algo,
        farm_results,
        point_results,
        extra_sig=extra_sig,
    )
    if output is None:
        raise ValueError(f"Output {index}: Could not construct {output_type}")
    if not type(output).__module__.startswith("foxes.models.outputs."):
        raise ValueError(
            f"Output {index}: Constructor-defined outputs must belong "
            "to foxes.models.outputs"
        )

    # Execute through the ordinary output lifecycle and publish pipeline results.
    output.nofig = nofig or output.nofig
    result = algo.calc_outputs(
        outputs=output, farm_results=farm_results, point_results=point_results
    )
    if labels is not None:
        _store_labels(results_storage, labels, result)
    if (
        isinstance(result, Dataset)
        and getattr(output, "operation", "").startswith("add_")
        and farm_results is not None
    ):
        farm_results.update(result)
    return result


def run_obj_function(
    obj: Any,
    fdict: Dict[Any, Any],
    algo: Algorithm | None,
    with_engine: bool,
    results_storage: dict[Any, Any],
    nofig: bool = False,
    verbosity: int | None = None,
) -> Any:
    """
    Run one configured object method and register its returned values.

    Parameters
    ----------
    obj
        Object exposing the configured method.
    fdict
        Mutable method-call configuration. Control keys are consumed and
        top-level ``$name`` arguments are replaced with stored result objects.
    algo
        Algorithm injected when the method declares ``algo`` or ``farm``.
    with_engine
        Whether execution is inside the engine context, used in progress output.
    results_storage
        Shared reference mapping, updated in place for requested result labels.
    nofig
        Suppress the ``plt_show`` and ``plt_close`` shortcuts in ``fdict``.
    verbosity
        Progress verbosity; ``0`` is silent and ``None`` prints all messages.

    Returns
    -------
    results
        Method return value, or ``None`` when the enabled ``plt_close`` shortcut
        closes its figure. Stored references point to this same value.

    Raises
    ------
    KeyError
        If the method name or a referenced result is missing.
    ValueError
        If an explicit result label is invalid.

    """

    def _print(*args: Any, level: int = 1, **kwargs: Any) -> None:
        if verbosity is None or verbosity >= level:
            print(*args, **kwargs)

    fname = fdict.pop_item("function")
    _print(f"Running function {type(obj).__name__}.{fname} (with_engine={with_engine})")
    plt_show = fdict.pop_item("plt_show", False)
    plt_close = fdict.pop_item("plt_close", False)
    rlbs = fdict.pop_item("result_labels", None)

    # grab function:
    ocls = type(obj).__name__
    assert hasattr(obj, fname), f"Output of type '{ocls}': Function '{fname}' not found"
    f = getattr(obj, fname)

    # add required input data objects:
    prs = list(signature(f).parameters.keys())
    if "algo" in prs:
        fdict["algo"] = algo
    if "farm" in prs:
        assert algo is not None, f"Output of type '{ocls}' requires algo"
        fdict["farm"] = algo.farm

    # replace result labels by objects:
    for k, d in fdict.items():
        if isinstance(d, str) and d[0] == "$":
            fdict[k] = _get_object(results_storage, d)

    # run function:
    args = fdict.pop_item("args", tuple())
    results = f(*args, **fdict)

    # pyplot shortcuts:
    if not nofig and plt_show:
        plt.show()
    if not nofig and plt_close:
        results = None
        plt.close()

    # store results under result labels:
    if rlbs is not None:
        _store_labels(results_storage, rlbs, results)

    return results


def run_outputs(
    idict: Dict[Any, Any],
    algo: Algorithm | None = None,
    farm_results: Dataset | None = None,
    point_results: Any = None,
    with_engine: bool = False,
    extra_sig: dict[str, Any] = {},
    results_storage: Dict[Any, Any] | None = None,
    ret_results_storage: bool = False,
    nofig: bool = False,
    verbosity: int | None = None,
) -> tuple[list[tuple[dict[str, Any], list[Any] | None]], Dict[Any, Any]]:
    """
    Run configured output models and object methods for one engine phase.

    Entries with ``output_type`` and no ``functions`` list construct output
    models and call ``algo.calc_outputs``. Other entries execute their configured
    method lists. Stored results can be referenced by later entries.

    Parameters
    ----------
    idict
        Input dictionary containing an ``outputs`` list. Entries are copied
        before consuming execution options or resolving references.
    algo
        Algorithm used for constructor injection and output execution.
    farm_results
        Supplied ``(FC.STATE, FC.TURBINE)`` farm fields. Constructor-defined
        ``add_*`` evaluations update this dataset for subsequent entries.
    point_results
        Supplied point results, forwarded without scientific recalculation when
        the selected output supports precomputed fields.
    with_engine
        Select entries or methods for this engine phase. Constructor-defined
        outputs default to execution inside the engine context; method entries
        default to execution outside it.
    extra_sig
        Parameter-name-to-value mapping injected into matching constructors.
    results_storage
        Shared result-reference mapping, updated in place. ``None`` creates a
        new mapping. Constructor options resolve nested ``$name`` references;
        configured methods resolve top-level references.
    ret_results_storage
        Request result-reference storage explicitly. The returned pair includes
        this storage for either setting.
    nofig
        Suppress output-model figures and configured pyplot show/close shortcuts.
    verbosity
        Progress verbosity; ``0`` is silent and ``None`` prints all messages.

    Returns
    -------
    outputs
        Ordered ``(definition, results)`` pairs. A constructor-defined output has
        one result; method entries have one value per configured call, including
        ``None`` for skipped calls. Skipped entries have ``results=None``.
    results_storage
        The supplied or newly created reference mapping; values are not copied.

    Raises
    ------
    KeyError
        If an entry specifies neither ``output_type`` nor ``object``, a required
        option is missing, or a result reference is unavailable.
    ValueError
        If a constructor-defined output lacks an algorithm, cannot be constructed,
        belongs outside ``foxes.models.outputs``, or uses invalid result labels.
    TypeError
        If the algorithm lacks ``calc_outputs`` for a constructor-defined entry.

    """

    def _print(*args: Any, level: int = 1, **kwargs: Any) -> None:
        if verbosity is None or verbosity >= level:
            print(*args, **kwargs)

    if results_storage is None:
        results_storage = Dict(_name="result_storage")

    out: list[tuple[dict[str, Any], list[Any] | None]] = []
    if "outputs" in idict:
        odicts = idict["outputs"]

        for i, d in enumerate(odicts):
            d = deepcopy(d)
            if "output_type" in d:
                if "functions" not in d:
                    # Preserve the input description before consuming model options.
                    definition = dict(d)
                    selected_engine = d.pop("with_engine", True)
                    if selected_engine != with_engine:
                        out.append((definition, None))
                        continue
                    result = _run_output_model(
                        i,
                        d,
                        algo,
                        farm_results,
                        point_results,
                        extra_sig,
                        results_storage,
                        nofig,
                    )
                    out.append((definition, [result]))
                    continue
                d["nofig"] = nofig
                ocls = d.pop_item("output_type")
                d0 = dict(output_type=ocls)
                d0.update(d)

                flist = d.pop_item("functions")
                ematch = [fd.pop("with_engine", False) == with_engine for fd in flist]

                if with_engine and any(ematch) and not all(ematch):
                    ecount = sum(ematch)
                    assert not any(ematch[ecount:]), (
                        f"Output {i}, {ocls}: with_engine is True "
                        f"but functions with with_engine=False are not at the end: {ematch}"
                    )

                if any(ematch) or (d.pop("with_engine", False) and with_engine):
                    o = get_output_obj(
                        ocls, d, algo, farm_results, point_results, extra_sig=extra_sig
                    )
                else:
                    o = None

            elif "object" in d:
                ocls = d.pop_item("object")
                d0 = dict(object=ocls)
                d0.update(d)

                flist = d.pop_item("functions")
                ematch = [fd.pop("with_engine", False) == with_engine for fd in flist]

                if with_engine and any(ematch) and not all(ematch):
                    ecount = sum(ematch)
                    assert not any(ematch[ecount:]), (
                        f"Output {i}, {ocls}: with_engine is True "
                        f"but functions with with_engine=False are not at the end: {ematch}"
                    )

                if any(ematch):
                    o = _get_object(results_storage, ocls)
                else:
                    o = None

            else:
                raise KeyError(
                    f"Output {i}: Please specify either 'output_type' or 'object'"
                )

            if o is None:
                out.append((d0, None))
            else:
                _print(f"Entering output {i}, {ocls} (with_engine={with_engine})")
                fres: list[Any | None] = []
                for fdict, em in zip(flist, ematch):
                    if em:
                        results = (
                            run_obj_function(
                                o,
                                fdict,
                                algo,
                                with_engine,
                                results_storage,
                                nofig,
                                verbosity,
                            )
                            if em
                            else None
                        )
                    else:
                        results = None
                    fres.append(results)
                out.append((d0, fres))

        if len(odicts):
            _print()

    return out if not ret_results_storage else out, results_storage


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
    """
    Runs foxes from dictionary input

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
        Parameters for engine generation, overrules idict
    iterative
        Add iterative algorithm wrapper, overrules idict
    nofig
        Do not show figures, overrules settings from idict
    verbosity
        Force a verbosity level, 0 = silent, overrules
        settings from idict
    work_dir
        The main working directory path
    input_dir
        The input directory path
    output_dir
        The output directory path
    algo_pars
        Additional parameters for read_dict

    Returns
    -------
    farm_results
        The farm results
    point_results
        The point results
    outputs
        For each output enty, a tuple (dict, results),
        where results is a list that represents one
        entry per function call


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
