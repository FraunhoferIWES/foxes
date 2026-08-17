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
from foxes.output import Output
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

    :group: input.yaml

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

    :group: input.yaml

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
    Runs a function of an object

    Parameters
    ----------
    obj
        The object
    fdict
        The function call dict
    algo
        The algorithm
    with_engine
        Flag for running from within engine context
    results_storage
        Storage for result variables
    nofig
        Do not show figures, overrules settings from fdict
    verbosity
        The verbosity level, 0 = silent

    Returns
    -------
    results
        The returns of the function

    :group: input.yaml

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

        def _set_label(results_storage: dict[Any, Any], k: str, r: Any) -> None:
            if k not in ["", "none", "None", "_", "__"]:
                assert k[0] == "$", (
                    f"Output of type '{ocls}', function '{fname}': result labels must start with '$', got '{k}'"
                )
                assert "[" not in k and "]" not in k and "," not in k, (
                    f"Output of type '{ocls}', function '{fname}': result labels cannot contain '[' or ']' or comma, got '{k}'"
                )
                _print(f"    result label {k}: {type(r).__name__}")
                results_storage[k] = r

        if isinstance(rlbs, (list, tuple)):
            for i, k in enumerate(rlbs):
                _set_label(results_storage, k, results[i])
        else:
            _set_label(results_storage, rlbs, results)

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
    Run outputs from dict.

    Parameters
    ----------
    engine
        The engine object
    idict
        The input parameter dictionary
    algo
        The algorithm
    farm_results
        The farm results
    point_results
        The point results
    with_engine
        Flag for running from within engine context
    extra_sig
        Extra function signature check, sets
        arguments (key) with data (value)
    results_storage
        Storage for result variables
    ret_results_storage
        Flag for returning results variables
    nofig
        Do not show figures, overrules settings from idict
    verbosity
        The verbosity level, 0 = silent

    Returns
    -------
    outputs
        For each output enty, a tuple (dict, results),
        where results is a list that represents one
        entry per function call
    results_storage
        The results variables

    :group: input.yaml

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

    :group: input.yaml

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
