import matplotlib.pyplot as plt
import pandas as pd
from inspect import signature

import foxes.input.farm_layout as farm_layout
from foxes.core import States, Engine, WindFarm, Algorithm
from foxes.models import ModelBook
from foxes.output import Output
from foxes.utils import Dict, new_cls
from foxes.config import config
import foxes.constants as FC


def read_dict(
    idict,
    farm=None,
    states=None,
    mbook=None,
    algo=None,
    engine_pars=None,
    iterative=None,
    verbosity=None,
    work_dir=".",
    out_dir=".",
    **algo_pars,
):
    """
    Read dictionary input into foxes objects

    Parameters
    ----------
    idict: foxes.utils.Dict
        The input parameter dictionary
    farm: foxes.core.WindFarm, optional
        The wind farm, overrules settings from idict
    states: foxes.core.States, optional
        The ambient states, overrules settings from idict
    mbook: foxes.models.ModelBook, optional
        The model book, overrules settings from idict
    algo: foxes.core.Algorithm, optional
        The algorithm, overrules settings from idict
    engine_pars: dict, optional
        Parameters for engine creation, overrules
        settings from idict
    iterative: bool, optional
        Force iterative calculations, overrules
        settings from idict
    verbosity: int, optional
        Force a verbosity level, 0 = silent, overrules
        settings from idict
    work_dir: str or pathlib.Path
        Path to the working directory
    out_dir: str or pathlib.Path
        The default output directory
    algo_pars: dict, optional
        Additional parameters for the algorithm, overrules
        settings from idict

    Returns
    -------
    algo: foxes.core.Algorithm
        The algorithm
    engine: foxes.core.Engine
        The engine, or None if not set

    :group: input.yaml

    """

    def _print(*args, level=1, **kwargs):
        if verbosity is None or verbosity >= level:
            print(*args, **kwargs)

    # set working directory:
    config[FC.WORK_DIR] = work_dir
    config[FC.OUT_DIR] = out_dir
    _print("Working directory:", config.work_dir)
    _print("Output directory :", config.out_dir)

    # create states:
    if states is None:
        _print("Creating states")
        states = States.new(**idict["states"])

    # create model book:
    if mbook is None:
        mbook = ModelBook()
        if "model_book" in idict:
            _print("Creating model book")
            mdict = idict.get_item("model_book")
            for s, mlst in mdict.items():
                t = mbook.sources.get_item(s)
                c = mbook.base_classes.get_item(s)
                ms = [Dict(m, name=f"{mdict.name}.s{i}") for i, m in enumerate(mlst)]
                for m in ms:
                    mname = m.pop_item("name")
                    _print(f"  Adding {s}.{mname}")
                    t[mname] = c.new(**m)

    # create farm:
    if farm is None:
        _print("Creating wind farm")
        fdict = idict.get_item("wind_farm")
        lyts = [
            Dict(l, name=f"{fdict.name}.layout{i}")
            for i, l in enumerate(fdict.pop_item("layouts"))
        ]
        farm = WindFarm(**fdict)
        for lyt in lyts:
            add_fun = getattr(farm_layout, lyt.pop_item("function"))
            if verbosity is not None:
                lyt["verbosity"] = verbosity - 1
            add_fun(farm, **lyt)

    # create engine:
    engine = None
    if engine_pars is not None:
        engine = Engine.new(**engine_pars)
        _print(f"Initializing engine: {engine}")
        engine.initialize()
    elif "engine" in idict:
        if verbosity is not None:
            idict["verbosity"] = verbosity - 1
        engine = Engine.new(**idict["engine"])
        engine.initialize()
        _print(f"Initializing engine: {engine}")

    # create algorithm:
    if algo is None:
        _print("Creating algorithm")
        adict = idict.get_item("algorithm")
        if iterative is not None and iterative:
            adict["algo_type"] = "Iterative"
        adict.update(dict(farm=farm, states=states, mbook=mbook))
        if verbosity is not None:
            adict["verbosity"] = verbosity - 1
        if algo_pars is not None:
            adict.update({v: d for v, d in algo_pars.items() if d is not None})
        algo = Algorithm.new(**adict)

    return algo, engine


def run_outputs(
    idict,
    algo=None,
    farm_results=None,
    point_results=None,
    verbosity=None,
):
    """
    Run outputs from dict.

    Parameters
    ----------
    idict: foxes.utils.Dict
        The input parameter dictionary
    algo: foxes.core.Algorithm, optional
        The algorithm
    farm_results: xarray.Dataset, optional
        The farm results
    point_results: xarray.Dataset, optional
        The point results
    verbosity: int, optional
        Force a verbosity level, 0 = silent, overrules
        settings from idict

    Returns
    -------
    outputs: list of tuple
        For each output enty, a tuple (dict, results),
        where results is a tuple that represents one
        entry per function call

    :group: input.yaml

    """

    def _print(*args, level=1, **kwargs):
        if verbosity is None or verbosity >= level:
            print(*args, **kwargs)

    out = []
    if "outputs" in idict:
        _print("Running outputs")
        odicts = [
            Dict(odict, name=f"{idict.name}.output{i}")
            for i, odict in enumerate(idict["outputs"])
        ]

        rlabels = Dict(name="result_labels")

        def _get_object(rlabels, d):
            d = d.replace("]", "")
            i0 = d.find("[")
            if i0 > 0:
                inds = tuple([int(x) for x in d[i0 + 1 :].split(",")])
                return rlabels[d[:i0]][inds]
            else:
                return rlabels[d]

        for i, d in enumerate(odicts):

            if "output_type" in d:
                ocls = d.pop_item("output_type")
                _print(f"  Output {i}: {ocls}")
                d0 = dict(output_type=ocls)
                d0.update(d)
                flist = [
                    Dict(f, name=f"{d.name}.function{i}")
                    for i, f in enumerate(d.pop_item("functions"))
                ]
                cls = new_cls(Output, ocls)
                prs = list(signature(cls.__init__).parameters.keys())
                if "algo" in prs:
                    assert (
                        algo is not None
                    ), f"Output {i} of type '{ocls}' requires algo"
                    d["algo"] = algo
                if "farm" in prs:
                    d["farm"] = algo.farm
                if "farm_results" in prs:
                    if farm_results is None:
                        print(f"No farm results; skipping output {ocls}")
                        for fdict in flist:
                            out += (d0, None)
                        continue
                    d["farm_results"] = farm_results
                if "point_results" in prs:
                    d["point_results"] = point_results
                o = cls(**d)

            elif "object" in d:
                ocls = d.pop("object")
                o = _get_object(rlabels, ocls)
                d0 = dict(object=ocls)
                d0.update(d)
                flist = [
                    Dict(f, name=f"{d.name}.function{i}")
                    for i, f in enumerate(d.pop_item("functions"))
                ]

            else:
                raise KeyError(
                    f"Output {i} of type '{ocls}': Please specify either 'output_type' or 'object'"
                )

            fres = []
            for fdict in flist:
                fname = fdict.pop_item("function")
                _print(f"  - {fname}")
                plt_show = fdict.pop("plt_show", False)
                plt_close = fdict.pop("plt_close", False)
                rlbs = fdict.pop("result_labels", None)

                # grab function:
                assert hasattr(
                    o, fname
                ), f"Output {i} of type '{ocls}': Function '{fname}' not found"
                f = getattr(o, fname)

                # add required input data objects:
                prs = list(signature(f).parameters.keys())
                if "algo" in prs:
                    fdict["algo"] = algo
                if "farm" in prs:
                    fdict["farm"] = algo.farm

                # replace result labels by objects:
                for k, d in fdict.items():
                    if isinstance(d, str) and d[0] == "$":
                        fdict[k] = _get_object(rlabels, d)

                # run function:
                args = fdict.pop("args", tuple())
                results = f(*args, **fdict)

                # pyplot shortcuts:
                if plt_show:
                    plt.show()
                if plt_close:
                    results = None
                    plt.close()

                # store results under result labels:
                if rlbs is not None:

                    def _set_label(rlabels, k, r):
                        if k not in ["", "none", "None", "_", "__"]:
                            assert (
                                k[0] == "$"
                            ), f"Output {i} of type '{ocls}', function '{fname}': result labels must start with '$', got '{k}'"
                            assert (
                                "[" not in k and "]" not in k and "," not in k
                            ), f"Output {i} of type '{ocls}', function '{fname}': result labels cannot contain '[' or ']' or comma, got '{k}'"
                            _print(f"    result label {k}: {type(r).__name__}")
                            rlabels[k] = r

                    if isinstance(rlbs, (list, tuple)):
                        for i, k in enumerate(rlbs):
                            _set_label(rlabels, k, results[i])
                    else:
                        _set_label(rlabels, rlbs, results)

                fres.append(results)
            out.append((d0, fres))

    return out


def run_dict(idict, *args, verbosity=None, **kwargs):
    """
    Runs foxes from dictionary input

    Parameters
    ----------
    idict: foxes.utils.Dict
        The input parameter dictionary
    args: tuple, optional
        Additional parameters for read_dict
    verbosity: int, optional
        Force a verbosity level, 0 = silent, overrules
        settings from idict
    kwargs: dict, optional
        Additional parameters for read_dict

    Returns
    -------
    farm_results: xarray.Dataset, optional
        The farm results
    point_results: xarray.Dataset, optional
        The point results
    outputs: list of tuple
        For each output enty, a tuple (dict, results),
        where results is a tuple that represents one
        entry per function call

    :group: input.yaml

    """

    def _print(*args, level=1, **kwargs):
        if verbosity is None or verbosity >= level:
            print(*args, **kwargs)

    # read components:
    algo, engine = read_dict(idict, *args, verbosity=verbosity, **kwargs)

    # run farm calculation:
    rdict = idict.get_item("calc_farm", Dict(name=idict.name + ".calc_farm"))
    if rdict.pop_item("run", True):
        _print("Running calc_farm")
        farm_results = algo.calc_farm(**rdict)
    else:
        farm_results = None
    out = (farm_results,)

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

    # run outputs:
    out += (run_outputs(idict, algo, farm_results, point_results, verbosity),)

    # shutdown engine, if created above:
    if engine is not None:
        _print(f"Finalizing engine: {engine}")
        engine.finalize()

    return out
