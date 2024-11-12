import matplotlib.pyplot as plt

import foxes.input.farm_layout as farm_layout
from foxes.core import States, Engine, WindFarm, Algorithm
from foxes.models import ModelBook
from foxes.output import Output
from foxes.utils import Dict


def run_dict(
        idict, 
        *args, 
        engine_pars=None, 
        iterative=None,
        verbosity=None, 
        **kwargs,
    ):
    """
    Runs foxes from dictionary input
    
    Parameters
    ----------
    idict: foxes.utils.Dict
        The input parameter dictionary
    engine_pars: dict, optional
        Parameters for engine creation
    iterative: bool, optional
        Force iterative calculations
    verbosity: int, optional
        Force a verbosity level, 0 = silent

    Returns
    -------


    :group: input.yaml
    
    """

    def _print(*args, level=1, **kwargs):
        if verbosity is None or verbosity >= level:
            print(*args, **kwargs)

    # create states:
    _print("Creating states")
    states = States.new(**idict["states"])

    # create model book:
    mbook = ModelBook()
    if "model_book" in idict:
        _print("Creating model book")
        mdict = idict.get_item("model_book")
        for s, mlst in mdict.items():
            t = mbook.sources.get_item(s)
            c = mbook.base_classes.get_item(s)
            ms = [Dict(m, name=f"{mdict.name}.s{i}")
                  for i, m in enumerate(mlst)]
            for m in ms:
                mname = m.pop_item("name")
                _print(f"  Adding {s}.{mname}")
                t[mname] = c.new(**m)

    # create farm:
    _print("Creating wind farm")
    fdict = idict.get_item("wind_farm")
    lyts = [Dict(l, name=f"{fdict.name}.layout{i}")
            for i, l in enumerate(fdict.pop_item("layouts"))]
    farm = WindFarm(**fdict)
    hverbo = 1 if verbosity is None else verbosity
    for lyt in lyts:
        add_fun = getattr(farm_layout, lyt.pop_item("function"))
        v = lyt.pop_item("verbosity", hverbo-1)
        add_fun(farm, verbosity=v, **lyt)
        
    # create engine:
    engine = None
    if engine_pars is not None:
        engine = Engine.new(**engine_pars)
        _print(f"Initializing engine: {engine}")
        engine.initialize()
    elif "engine" in idict:
        v = idict.pop_item("verbosity", hverbo)
        engine = Engine.new(**idict["engine"], verbosity=v)
        engine.initialize()
        _print(f"Initializing engine: {engine}")

    # create algorithm:
    _print("Creating algorithm")
    adict = idict.get_item("algorithm")
    if iterative is not None and iterative:
        adict["algo_type"] = "Iterative"
    adict.update(dict(farm=farm, states=states, mbook=mbook))
    v = adict.pop("verbosity", hverbo-1)
    algo = Algorithm.new(**adict, verbosity=v)

    # run farm calculation:
    rdict = idict.get_item("calc_farm")
    if rdict.pop_item("run"):
        _print("Running calc_farm")
        farm_results = algo.calc_farm(**rdict)
    else:
        farm_results = None

    # run outputs:
    out = (farm_results,)
    if "outputs" in idict:
        _print("Running outputs")
        odict = idict["outputs"]
        for ocls, d in odict.items():
            _print(f"  Output {ocls}")
            if d.pop_item("farm_results", False):
                d["farm_results"] = farm_results
            if d.pop_item("algo", False):
                d["algo"] = algo
            flist = [Dict(f, name=f"{d.name}.function{i}")
                     for i, f in enumerate(d.pop_item("functions"))]
            o = Output.new(ocls, **d)
            for fdict in flist:
                fname = fdict.pop_item("name")
                _print(f"  - {fname}")
                if fdict.pop_item("algo", False):
                    fdict["algo"] = algo
                plt_show = fdict.pop("plt_show", False)
                f = getattr(o, fname)
                res = f(**fdict)
                out += (res,) if not isinstance(res, tuple) else res
                if plt_show:
                    plt.show()

    # shutdown engine, if created above:
    if engine is not None:
        _print(f"Finalizing engine: {engine}")
        engine.finalize()
    
    return out
    