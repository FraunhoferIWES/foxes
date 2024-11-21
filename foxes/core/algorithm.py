import numpy as np
import xarray as xr
from abc import abstractmethod

from .model import Model
from foxes.data import StaticData
from foxes.utils import Dict, all_subclasses
from foxes.config import config
import foxes.constants as FC

from .engine import Engine


class Algorithm(Model):
    """
    Abstract base class for algorithms.

    Algorithms collect required objects for running
    calculations, and contain the calculation functions
    which are meant to be called from top level code.

    Attributes
    ----------
    verbosity: int
        The verbosity level, 0 means silent

    :group: core

    """

    def __init__(
        self,
        mbook,
        farm,
        verbosity=1,
        dbook=None,
        **engine_pars,
    ):
        """
        Constructor.

        Parameters
        ----------
        mbook: foxes.models.ModelBook
            The model book
        farm: foxes.WindFarm
            The wind farm
        verbosity: int
            The verbosity level, 0 means silent
        dbook: foxes.DataBook, optional
            The data book, or None for default
        engine_pars: dict, optional
            Parameters for the engine constructor

        """
        super().__init__()

        self.name = type(self).__name__
        self.verbosity = verbosity
        self.n_states = None
        self.n_turbines = farm.n_turbines

        self.__farm = farm
        self.__mbook = mbook
        self.__dbook = StaticData() if dbook is None else dbook
        self.__idata_mem = Dict(name="idata_mem")
        self.__chunk_store = Dict(name="chunk_store")

        if len(engine_pars):
            if "engine_type" in engine_pars:
                if "engine" in engine_pars:
                    raise KeyError(
                        f"{self.name}: Expecting either 'engine' or 'engine_type', not both"
                    )
            elif "engine" in engine_pars:
                engine_pars["engine_type"] = engine_pars.pop("engine")
            v = engine_pars.pop("verbosity", verbosity)
            try:
                e = Engine.new(verbosity=v, **engine_pars)
            except TypeError as e:
                print(f"\nError while interpreting engine_pars {engine_pars}\n")
                raise e
            self.print(f"Algorithm '{self.name}': Selecting engine '{e}'")
            e.initialize()

    @property
    def farm(self):
        """
        The wind farm

        Returns
        -------
        mb: foxes.core.WindFarm
            The wind farm

        """
        return self.__farm

    @property
    def mbook(self):
        """
        The model book

        Returns
        -------
        mb: foxes.models.ModelBook()
            The model book

        """
        if self.running:
            raise ValueError(
                f"Algorithm '{self.name}': Cannot access mbook while running"
            )
        return self.__mbook

    @property
    def dbook(self):
        """
        The data book

        Returns
        -------
        mb: foxes.data.StaticData()
            The data book

        """
        if self.running:
            raise ValueError(
                f"Algorithm '{self.name}': Cannot access dbook while running"
            )
        return self.__dbook

    @property
    def idata_mem(self):
        """
        The current idata memory

        Returns
        -------
        dict :
            Keys: model name, value: idata dict

        """
        if self.running:
            raise ValueError(
                f"Algorithm '{self.name}': Cannot access idata_mem while running"
            )
        return self.__idata_mem

    @property
    def chunk_store(self):
        """
        The current chunk store

        Returns
        -------
        dict :
            Keys: model name, value: idata dict

        """
        return self.__chunk_store

    def print(self, *args, vlim=1, **kwargs):
        """
        Print function, based on verbosity.

        Parameters
        ----------
        args: tuple, optional
            Arguments for the print function
        kwargs: dict, optional
            Keyword arguments for the print function
        vlim: int
            The verbosity limit

        """
        if self.verbosity >= vlim:
            print(*args, **kwargs)

    def initialize(self):
        """
        Initializes the algorithm.
        """
        if self.running:
            raise ValueError(
                f"Algorithm '{self.name}': Cannot initialize while running"
            )
        super().initialize(self, self.verbosity - 1)

    def store_model_data(self, model, idata, force=False):
        """
        Store model data

        Parameters
        ----------
        model: foxes.core.Model
            The model
        idata: dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`
        force: bool
            Overwrite existing data

        """
        mname = f"{type(model).__name__}_{model.name}"
        if force:
            self.__idata_mem[mname] = idata
        elif mname in self.idata_mem:
            raise KeyError(f"Attempt to overwrite stored data for model '{mname}'")
        else:
            self.idata_mem[mname] = idata

    def get_model_data(self, model):
        """
        Gets model data from memory

        Parameters
        ----------
        model: foxes.core.Model
            The model

        """
        mname = f"{type(model).__name__}_{model.name}"
        try:
            return self.idata_mem[mname]
        except KeyError:
            raise KeyError(
                f"Key '{mname}' not found in idata_mem, available keys: {sorted(list(self.idata_mem.keys()))}"
            )

    def del_model_data(self, model):
        """
        Remove stored model data

        Parameters
        ----------
        model: foxes.core.Model
            The model

        """
        mname = f"{type(model).__name__}_{model.name}"
        try:
            del self.idata_mem[mname]
        except KeyError:
            raise KeyError(f"Attempt to delete data of model '{mname}', but not stored")

    def update_n_turbines(self):
        """
        Reset the number of turbines,
        according to self.farm
        """
        if self.n_turbines != self.farm.n_turbines:
            self.n_turbines = self.farm.n_turbines

            # resize stored idata, if dependent on turbine coord:
            newk = {}
            for mname, idata in self.idata_mem.items():
                if mname[:2] == "__":
                    continue
                for dname, d in idata["data_vars"].items():
                    k = f"__{mname}_{dname}_turbinv"
                    if k in self.idata_mem:
                        ok = self.idata_mem[k]
                    else:
                        ok = None
                        if FC.TURBINE in d[0]:
                            i = d[0].index(FC.TURBINE)
                            ok = np.unique(d[1], axis=1).shape[i] == 1
                        newk[k] = ok
                    if ok is not None:
                        if not ok:
                            raise ValueError(
                                f"{self.name}: Stored idata entry '{mname}:{dname}' is turbine dependent, unable to reset n_turbines"
                            )
                        if FC.TURBINE in idata["coords"]:
                            idata["coords"][FC.TURBINE] = np.arange(self.n_turbines)
                        i = d[0].index(FC.TURBINE)
                        n0 = d[1].shape[i]
                        if n0 > self.n_turbines:
                            idata["data_vars"][dname] = (
                                d[0],
                                np.take(d[1], range(self.n_turbines), axis=i),
                            )
                        elif n0 < self.n_turbines:
                            shp = [
                                d[1].shape[j] if j != i else self.n_turbines - n0
                                for j in range(len(d[1].shape))
                            ]
                            a = np.zeros(shp, dtype=d[1].dtype)
                            shp = [
                                d[1].shape[j] if j != i else 1
                                for j in range(len(d[1].shape))
                            ]
                            a[:] = np.take(d[1], -1, axis=i).reshape(shp)
                            idata["data_vars"][dname] = (
                                d[0],
                                np.append(d[1], a, axis=i),
                            )

            self.idata_mem.update(newk)

    def get_models_idata(self):
        """
        Returns idata object of models

        Returns
        -------
        idata: dict, optional
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`.
            Take algorithm's idata object by default.

        """
        if not self.initialized:
            raise ValueError(
                f"Algorithm '{self.name}': get_models_idata called before initialization"
            )
        idata = {"coords": {}, "data_vars": {}}
        for k, hidata in self.idata_mem.items():
            if len(k) < 3 or k[:2] != "__":
                idata["coords"].update(hidata["coords"])
                idata["data_vars"].update(hidata["data_vars"])
        return idata

    def get_models_data(self, idata=None, sel=None, isel=None):
        """
        Creates xarray from model input data.

        Parameters
        ----------
        idata: dict, optional
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`.
            Take algorithm's idata object by default.
        sel: dict, optional
            Selection of coordinates in dataset
        isel: dict, optional
            Selection of coordinates in dataset

        Returns
        -------
        ds: xarray.Dataset
            The model input data

        """
        if idata is None:
            idata = self.get_models_idata()
        ds = xr.Dataset(**idata)
        if isel is not None:
            ds = ds.isel(isel)
        if sel is not None:
            ds = ds.sel(sel)
        return ds

    def new_point_data(self, points, states_indices=None, n_states=None):
        """
        Creates a point data xarray object, containing only points.

        Parameters
        ----------
        points: numpy.ndarray
            The points, shape: (n_states, n_points, 3)
        states_indices: array_like, optional
            The indices of the states dimension
        n_states: int, optional
            The number of states

        Returns
        -------
        xarray.Dataset
            A dataset containing the points data

        """
        if n_states is None:
            n_states = self.n_states
        if states_indices is None:
            idata = {"coords": {}, "data_vars": {}}
        else:
            idata = {"coords": {FC.STATE: states_indices}, "data_vars": {}}

        if len(points.shape) == 2 and points.shape[1] == 3:
            pts = np.zeros((n_states,) + points.shape, dtype=config.dtype_double)
            pts[:] = points[None]
            points = pts
            del pts

        if (
            len(points.shape) != 3
            or points.shape[0] != n_states
            or points.shape[2] != 3
        ):
            raise ValueError(
                f"points have wrong dimensions, expecting ({n_states}, {points.shape[1]}, 3), got {points.shape}"
            )
        idata["data_vars"][FC.TARGETS] = (
            (FC.STATE, FC.TARGET, FC.TPOINT, FC.XYH),
            points[:, :, None, :],
        )
        idata["data_vars"][FC.TWEIGHTS] = (
            (FC.TPOINT,),
            np.array([1.0], dtype=config.dtype_double),
        )

        return xr.Dataset(**idata)

    def find_chunk_in_store(
        self,
        mdata,
        tdata=None,
        prev_s=0,
        prev_t=0,
        error=True,
    ):
        """
        Finds indices in chunk store

        Parameters
        ----------
        name: str
            The data name
        mdata: foxes.core.MData
            The mdata object
        tdata: foxes.core.TData, optional
            The tdata object
        prev_s: int
            How many states chunks backward
        prev_t: int
            How many points chunks backward
        error: bool
            Flag for raising KeyError if data not found

        Returns
        -------
        inds: tuple
            The (i0, n_states, t0, n_targets) data of the
            returning chunk

        """
        i0 = int(mdata.states_i0(counter=True))
        t0 = int(tdata.targets_i0() if tdata is not None else 0)
        n_states = int(mdata.n_states)
        n_targets = int(tdata.n_targets if tdata is not None else 0)

        if prev_s > 0 or prev_t > 0:

            inds = np.array(
                [
                    [
                        d["i0"],
                        d["i0"] + d["n_states"],
                        d["n_states"],
                        d["t0"],
                        d["t0"] + d["n_targets"],
                        d["n_targets"],
                    ]
                    for d in self.chunk_store.values()
                ],
                dtype=int,
            )

            if prev_t > 0:
                while prev_t > 0:
                    sel = np.where((inds[:, 0] == i0) & (inds[:, 4] == t0))[0]
                    if len(sel) == 0:
                        if error:
                            raise KeyError(
                                f"{self.name}: Previous key {(i0, t0)}, prev={(prev_s, prev_t)}, not found in chunk store, got inds {inds}"
                            )
                        else:
                            return None
                    else:
                        n_targets = inds[sel[0], 5]
                        t0 -= n_targets
                        prev_t -= 1

            if prev_s > 0:
                while prev_s > 0:
                    sel = np.where((inds[:, 1] == i0) & (inds[:, 3] == t0))[0]
                    if len(sel) == 0:
                        if error:
                            raise KeyError(
                                f"{self.name}: Previous key {(i0, t0)}, prev={(prev_s, prev_t)}, not found in chunk store, got inds {inds}"
                            )
                        else:
                            return None
                    else:
                        n_states = inds[sel[0], 2]
                        i0 -= n_states
                        prev_s -= 1

        return i0, n_states, t0, n_targets

    def add_to_chunk_store(
        self,
        name,
        data,
        mdata,
        tdata=None,
        copy=True,
    ):
        """
        Add data to the chunk store

        Parameters
        ----------
        name: str
            The data name
        data: numpy.ndarray
            The data
        mdata: foxes.core.MData
            The mdata object
        tdata: foxes.core.TData, optional
            The tdata object
        copy: bool
            Flag for copying incoming data

        """
        i0 = int(mdata.states_i0(counter=True))
        t0 = int(tdata.targets_i0() if tdata is not None else 0)

        key = (i0, t0)
        if key not in self.chunk_store:
            n_states = int(mdata.n_states)
            n_targets = int(tdata.n_targets if tdata is not None else 0)
            self.chunk_store[key] = Dict(
                {
                    "i0": i0,
                    "t0": t0,
                    "n_states": n_states,
                    "n_targets": n_targets,
                },
                name=f"chunk_store_{i0}_{t0}",
            )

        self.chunk_store[key][name] = data.copy() if copy else data

    def get_from_chunk_store(
        self,
        name,
        mdata,
        tdata=None,
        prev_s=0,
        prev_t=0,
        ret_inds=False,
        error=True,
    ):
        """
        Get data to the chunk store

        Parameters
        ----------
        name: str
            The data name
        mdata: foxes.core.MData
            The mdata object
        tdata: foxes.core.TData, optional
            The tdata object
        prev_s: int
            How many states chunks backward
        prev_t: int
            How many points chunks backward
        ret_inds: bool
            Also return (i0, n_states, t0, n_targets)
            of the returned chunk
        error: bool
            Flag for raising KeyError if data not found

        Returns
        -------
        data: numpy.ndarray
            The data
        inds: tuple, optional
            The (i0, n_states, t0, n_targets) data of the
            returning chunk

        """
        inds = self.find_chunk_in_store(mdata, tdata, prev_s, prev_t, error)

        if inds is None:
            return (None, (None, None, None, None)) if ret_inds else None
        else:
            i0, __, t0, __ = inds
            try:
                data = self.chunk_store[(i0, t0)][name]
            except KeyError as e:
                if error:
                    raise e
                else:
                    data = None
            if ret_inds:
                return data, inds
            else:
                return data

    def reset_chunk_store(self, new_chunk_store=None):
        """
        Resets the chunk store

        Parameters
        ----------
        new_chunk_store: foxes.utils.Dict, optional
            The new chunk store

        Returns
        -------
        chunk_store: foxes.utils.Dict
            The chunk store before resetting

        """
        chunk_store = self.chunk_store
        if new_chunk_store is None:
            self.__chunk_store = Dict(name="chunk_store")
        elif isinstance(new_chunk_store, Dict):
            self.__chunk_store = new_chunk_store
        else:
            self.__chunk_store = Dict(name="chunk_store")
            self.__chunk_store.update(new_chunk_store)
        return chunk_store

    def block_convergence(self, **kwargs):
        """
        Switch on convergence block during iterative run

        Parameters
        ----------
        kwargs: dict, optional
            Parameters for add_to_chunk_store()

        """
        self.add_to_chunk_store(
            name=FC.BLOCK_CONVERGENCE, data=True, copy=False, **kwargs
        )

    def eval_conv_block(self):
        """
        Evaluate convergence block, removing blocks on the fly

        Returns
        -------
        blocked: bool
            True if convergence is currently blocked

        """
        blocked = False
        for c in self.__chunk_store.values():
            blocked = c.pop(FC.BLOCK_CONVERGENCE, False) or blocked
        return blocked

    def set_running(
        self,
        algo,
        data_stash,
        sel=None,
        isel=None,
        verbosity=0,
    ):
        """
        Sets this model status to running, and moves
        all large data to stash.

        The stashed data will be returned by the
        unset_running() function after running calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data_stash: dict
            Large data stash, this function adds data here.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        assert algo is self

        super().set_running(algo, data_stash, sel, isel, verbosity)

        data_stash[self.name].update(
            dict(
                mbook=self.__mbook,
                dbook=self.__dbook,
                idata_mem=self.__idata_mem,
            )
        )
        del self.__mbook, self.__dbook
        self.__idata_mem = {}

    def unset_running(
        self,
        algo,
        data_stash,
        sel=None,
        isel=None,
        verbosity=0,
    ):
        """
        Sets this model status to not running, recovering large data
        from stash

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data_stash: dict
            Large data stash, this function adds data here.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        assert algo is self

        super().unset_running(algo, data_stash, sel, isel, verbosity)

        data = data_stash[self.name]
        self.__mbook = data.pop("mbook")
        self.__dbook = data.pop("dbook")
        self.__idata_mem = data.pop("idata_mem")

    @abstractmethod
    def _launch_parallel_farm_calc(
        self,
        *args,
        mbook,
        dbook,
        chunk_store,
        **kwargs,
    ):
        """
        Runs the main farm calculation, launching parallelization

        Parameters
        ----------
        args: tuple, optional
            Additional parameters for running
        mbook: foxes.models.ModelBook
            The model book
        dbook: foxes.DataBook
            The data book, or None for default
        chunk_store: foxes.utils.Dict
            The chunk store
        kwargs: dict, optional
            Additional parameters for running

        Returns
        -------
        farm_results: xarray.Dataset
            The farm results. The calculated variables have
            dimensions (state, turbine)

        """
        pass

    def calc_farm(self, *args, **kwargs):
        """
        Calculate farm data.

        Parameters
        ----------
        args: tuple, optional
            Parameters
        kwargs: dict, optional
            Keyword parameters

        Returns
        -------
        farm_results: xarray.Dataset
            The farm results. The calculated variables have
            dimensions (state, turbine)

        """
        if self.running:
            raise ValueError(
                f"Algorithm '{self.name}': Cannot call calc_farm while running"
            )

        # set to running:
        data_stash = {}
        chunk_store = self.reset_chunk_store()
        mdls = [
            m
            for m in [self] + list(args) + list(kwargs.values())
            if isinstance(m, Model)
        ]
        for m in mdls:
            m.set_running(
                self, data_stash, sel=None, isel=None, verbosity=self.verbosity - 2
            )

        # run parallel calculation:
        farm_results = self._launch_parallel_farm_calc(
            *args,
            chunk_store=chunk_store,
            sel=None,
            isel=None,
            **kwargs,
        )

        # reset to not running:
        for m in mdls:
            m.unset_running(
                self, data_stash, sel=None, isel=None, verbosity=self.verbosity - 2
            )

        return farm_results

    @abstractmethod
    def _launch_parallel_points_calc(
        self,
        *args,
        chunk_store,
        **kwargs,
    ):
        """
        Runs the main points calculation, launching parallelization

        Parameters
        ----------
        args: tuple, optional
            Additional parameters for running
        chunk_store: foxes.utils.Dict
            The chunk store
        kwargs: dict, optional
            Additional parameters for running

        Returns
        -------
        point_results: xarray.Dataset
            The point results. The calculated variables have
            dimensions (state, point)

        """
        pass

    def calc_points(self, *args, sel=None, isel=None, **kwargs):
        """
        Calculate points data.

        Parameters
        ----------
        args: tuple, optional
            Parameters
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        kwargs: dict, optional
            Keyword parameters

        Returns
        -------
        point_results: xarray.Dataset
            The point results. The calculated variables have
            dimensions (state, point)

        """
        if self.running:
            raise ValueError(
                f"Algorithm '{self.name}': Cannot call calc_points while running"
            )

        # set to running:
        data_stash = {}
        self.set_running(
            self, data_stash, sel=sel, isel=isel, verbosity=self.verbosity - 2
        )

        # run parallel calculation:
        chunk_store = self.reset_chunk_store()
        point_results = self._launch_parallel_points_calc(
            *args,
            chunk_store=chunk_store,
            sel=sel,
            isel=isel,
            **kwargs,
        )

        # reset to not running:
        self.unset_running(
            self, data_stash, sel=sel, isel=isel, verbosity=self.verbosity - 2
        )

        return point_results

    def finalize(self, clear_mem=False):
        """
        Finalizes the algorithm.

        Parameters
        ----------
        clear_mem: bool
            Clear idata memory

        """
        if self.running:
            raise ValueError(f"Algorithm '{self.name}': Cannot finalize while running")
        super().finalize(self, self.verbosity - 1)
        if clear_mem:
            self.__idata_mem = Dict()
            # self.reset_chunk_store()

    @classmethod
    def new(cls, algo_type, *args, **kwargs):
        """
        Run-time algorithm factory.

        Parameters
        ----------
        algo_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for the constructor
        kwargs: dict, optional
            Additional parameters for the constructor

        """

        if algo_type is None:
            return None

        allc = all_subclasses(cls)
        found = algo_type in [scls.__name__ for scls in allc]

        if found:
            for scls in allc:
                if scls.__name__ == algo_type:
                    return scls(*args, **kwargs)

        else:
            estr = (
                "Algorithm type '{}' is not defined, available types are \n {}".format(
                    algo_type, sorted([i.__name__ for i in allc])
                )
            )
            raise KeyError(estr)
