import matplotlib.pyplot as plt
import pandas as pd
from inspect import signature

import foxes.input.farm_layout as farm_layout
from foxes.core import States, Engine, WindFarm, Algorithm
from foxes.models import ModelBook
from foxes import output
from foxes.utils import Dict
from foxes.config import config
import foxes.constants as FC


def run_dict(
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
    Runs foxes from dictionary input

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
    farm_results: xarray.Dataset, optional
        The farm results
    point_results: xarray.Dataset, optional
        The point results
    output_i: object
        For each output either None or the output result

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
            adict.update(algo_pars)
        algo = Algorithm.new(**adict)

    # run farm calculation:
    rdict = idict.get_item("calc_farm")
    if rdict.pop_item("run"):
        _print("Running calc_farm")
        farm_results = algo.calc_farm(**rdict)
    else:
        farm_results = None
    out = (farm_results,)

    # run points calculation:
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
    if "outputs" in idict:
        _print("Running outputs")
        odict = idict["outputs"]
        for ocls, d in odict.items():
            _print(f"  Output {ocls}")
            flist = [
                Dict(f, name=f"{d.name}.function{i}")
                for i, f in enumerate(d.pop_item("functions"))
            ]
            try:
                cls = getattr(output, ocls)
            except AttributeError as e:
                print(f"\nClass '{ocls}' not found in outputs. Found:")
            prs = list(signature(cls.__init__).parameters.keys())
            if "algo" in prs:
                d["algo"] = algo
            if "farm_results" in prs:
                if farm_results is None:
                    print(f"No farm results; skipping output {ocls}")
                    for fdict in flist:
                        out += (None,)
                    continue
                d["farm_results"] = farm_results
            o = cls(**d)
            for fdict in flist:
                fname = fdict.pop_item("name")
                _print(f"  - {fname}")
                plt_show = fdict.pop("plt_show", False)
                f = getattr(o, fname)
                prs = list(signature(f).parameters.keys())
                if "algo" in prs:
                    fdict["algo"] = algo
                res = f(**fdict)
                out += (res,) if not isinstance(res, tuple) else res
                if plt_show:
                    plt.show()
                    plt.close()

    # shutdown engine, if created above:
    if engine is not None:
        _print(f"Finalizing engine: {engine}")
        engine.finalize()

    return out
