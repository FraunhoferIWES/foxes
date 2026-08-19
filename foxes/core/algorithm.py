from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
import xarray as xr
from typing import TYPE_CHECKING, Any, cast

from foxes.data import StaticData
from foxes.utils import DataBook, Dict, new_instance
from foxes.config import config
import foxes.constants as FC

from .engine import launch_parallel_calc
from .model import Model

if TYPE_CHECKING:
    from foxes.core.data import MData, TData
    from foxes.core.model import LoadedData
    from foxes.core.rotor_model import RotorModel
    from foxes.core.states import States
    from foxes.core.wind_farm import WindFarm
    from foxes.core.wake_frame import WakeFrame
    from foxes.core.wake_deflection import WakeDeflection
    from foxes.core.wake_model import WakeModel
    from foxes.core.partial_wakes_model import PartialWakesModel
    from foxes.core.ground_model import GroundModel
    from foxes.core.farm_controller import FarmController
    from foxes.models import ModelBook


class Algorithm(Model):
    """
    Abstract base class for algorithms.

    Algorithms collect required objects for running
    calculations, and contain the calculation functions
    which are meant to be called from top level code.

    Attributes
    ----------
    verbosity
        The verbosity level, 0 means silent


    """

    def __init__(
        self,
        mbook: ModelBook,
        farm: WindFarm,
        verbosity: int = 1,
        dbook: DataBook | None = None,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        mbook
            The model book.
        farm
            The wind farm.
        verbosity
            The verbosity level; ``0`` means silent.
        dbook
            The data book, or ``None`` for the default data book.

        """
        super().__init__()

        self.name = type(self).__name__
        self.verbosity = verbosity
        self.n_states: int | None = None
        self.n_turbines = farm.n_turbines

        self.__farm = farm
        self.__mbook = mbook
        self.__dbook = StaticData() if dbook is None else dbook
        self.__chunk_store: Dict[tuple[int, int], dict[str, Any]] = Dict(
            _name="chunk_store"
        )
        self.__loaded_data: LoadedData | None = None
        self.__farm_vars: list[str] = []

    @property
    def farm(self) -> WindFarm:
        """
        The wind farm

        Returns
        -------
        farm
            The wind farm

        """
        return self.__farm

    @property
    def mbook(self) -> ModelBook:
        """
        The model book

        Returns
        -------
        mbook
            The model book

        """
        if self.running:
            raise ValueError(
                f"Algorithm '{self.name}': Cannot access mbook while running"
            )
        return self.__mbook

    @property
    def dbook(self) -> DataBook:
        """
        The data book

        Returns
        -------
        dbook
            The data book

        """
        if self.running:
            raise ValueError(
                f"Algorithm '{self.name}': Cannot access dbook while running"
            )
        return self.__dbook

    @property
    def chunk_store(self) -> Dict[tuple[int, int], dict[str, Any]]:
        """
        The current chunk store

        Returns
        -------
        chunk_store
            Keys: chunk-state/point tuple, value: idata dict

        """
        return self.__chunk_store

    @property
    def states(self) -> States:
        """Ambient states model, provided by concrete algorithms."""
        raise ValueError(f"Algorithm '{self.name}': states model not available")

    @property
    def rotor_model(self) -> RotorModel:
        """Rotor model, provided by concrete algorithms."""
        raise ValueError(f"Algorithm '{self.name}': rotor_model not available")

    @property
    def wake_frame(self) -> WakeFrame:
        """Wake frame model, provided by concrete algorithms."""
        raise ValueError(f"Algorithm '{self.name}': wake_frame not available")

    @property
    def wake_deflection(self) -> WakeDeflection | None:
        """Wake deflection model, optional for algorithms without wake calculations."""
        return None

    @property
    def wake_models(self) -> dict[str, WakeModel]:
        """Wake model mapping by name."""
        return {}

    @property
    def partial_wakes(self) -> dict[str, PartialWakesModel]:
        """Partial wakes model mapping by wake-model name."""
        return {}

    @property
    def ground_models(self) -> dict[str, GroundModel]:
        """Ground model mapping by wake-model name."""
        return {}

    @property
    def farm_controller(self) -> FarmController:
        """Farm controller, provided by concrete algorithms."""
        raise ValueError(f"Algorithm '{self.name}': farm_controller not available")

    @property
    def max_wake_length_km(self) -> float:
        """Maximum wake length in km."""
        raise KeyError(f"Algorithm '{self.name}': No maximum wake length set")

    @property
    def has_max_wake_length(self) -> bool:
        """Whether a maximum wake length is configured."""
        return False

    @property
    def farm_vars(self) -> list[str]:
        """Farm output variable names produced by the algorithm."""
        return self.__farm_vars

    @farm_vars.setter
    def farm_vars(self, values: list[str]) -> None:
        self.__farm_vars = list(values)

    def chunked(self, data: xr.Dataset) -> xr.Dataset:
        """Optional hook for returning chunked results datasets."""
        return data

    @classmethod
    def get_model(cls, name: str) -> Any:
        """Return algorithm-specific helper model class by name."""
        raise NotImplementedError(
            f"Algorithm '{cls.__name__}': get_model is not implemented"
        )

    def _collect_point_models(
        self,
        calc_parameters: dict[str, dict[str, Any]] | None = None,
        point_models: Any = None,
        ambient: bool = False,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Collect the point-data model list for downstream point calculations."""
        raise NotImplementedError(
            f"Algorithm '{self.name}': _collect_point_models is not implemented"
        )

    @property
    def farm_results_downwind(self) -> xr.Dataset | None:
        """Previous-iteration farm results in downwind turbine order, if available."""
        return None

    @property
    def prev_farm_results(self) -> xr.Dataset | None:
        """Farm results from the previous iteration, if available."""
        return None

    @property
    def final_iteration(self) -> bool:
        """Whether the algorithm currently performs a final iteration pass."""
        return False

    @property
    def loaded_data(self) -> LoadedData:
        """
        The data loaded during initialization.

        Returns
        -------
        loaded_data
            The loaded data, containing the keys "coords", "data_vars", and
            "extra_data".

        """
        if self.__loaded_data is None:
            self.__loaded_data = self._empty_loaded_data()
        return self.__loaded_data

    def _empty_loaded_data(self) -> LoadedData:
        return {"coords": {}, "data_vars": {}, "extra_data": {}}

    def clear_loaded_data(self) -> None:
        """
        Clear the loaded data.

        Returns
        -------
        None

        """
        self.__loaded_data = self._empty_loaded_data()

    def get_model_data(self, pop: bool = False) -> tuple[xr.Dataset, dict[str, Any]]:
        """
        Get the model data.

        Parameters
        ----------
        pop
            Pop the model data from loaded_data

        Returns
        -------
        model_data
            The model data, containing all coords and data_vars from loaded_data
        extra_data
            The extra data from loaded_data

        """
        ld = self.loaded_data
        ed = ld["extra_data"]
        if pop:
            self.__loaded_data = self._empty_loaded_data()
        return xr.Dataset(coords=ld["coords"], data_vars=ld["data_vars"]), ed

    def print(self, *args: Any, vlim: int = 1, **kwargs: Any) -> None:
        """
        Print output based on the configured verbosity.

        Parameters
        ----------
        args
            Positional arguments for the print function.
        kwargs
            Keyword arguments for the print function.
        vlim
            The verbosity threshold for printing.

        """
        if self.verbosity >= vlim:
            print(*args, **kwargs)

    def print_deco(
        self,
        func_name: str | None = None,
        n_points: int | None = None,
    ) -> None:
        """
        Helper function for printing model names

        Parameters
        ----------
        func_name
            Name of the calling function
        n_points
            The number of points

        """
        if self.verbosity > 0:
            deco = "-" * 60
            print(f"\n{deco}")
            print(f"  Algorithm: {type(self).__name__}")
            if func_name is not None:
                print(f"  Running {self.name}: {func_name}")
            print(deco)
            print(f"  n_states : {self.n_states}")
            print(f"  n_turbines: {self.n_turbines}")

    def initialize(self, force: bool = False) -> None:
        """
        Initializes the algorithm.

        Parameters
        ----------
        force
            Overwrite existing data

        """
        if self.running:
            raise ValueError(
                f"Algorithm '{self.name}': Cannot initialize while running"
            )

        self.__loaded_data = super().initialize(
            algo=self,
            loaded_data=self.__loaded_data,
            force=force,
            verbosity=self.verbosity - 1,
        )

    def update_n_turbines(self) -> None:
        """
        Reset the number of turbines,
        according to self.farm
        """
        raise NotImplementedError()
        """
        if self.n_turbines != self.farm.n_turbines:
            self.n_turbines = self.farm.n_turbines

            # resize stored idata, if dependent on turbine coord:
            newk = {}
            for mname, idata in self.idata_mem.items():
                if mname[:2] == "__":
                    continue
                for dname, d in idata["data_vars"].items():
                    k = f"__{mname}_{dname}_turbine"
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
            """

    def new_point_data(
        self,
        points: np.ndarray,
        states_indices: Any = None,
        n_states: int | None = None,
    ) -> xr.Dataset:
        """
        Creates a point data xarray object, containing only points.

        Parameters
        ----------
        points
            The points, shape: (n_states, n_points, 3)
        states_indices
            The indices of the states dimension
        n_states
            The number of states

        Returns
        -------
        xarray.Dataset
            A dataset containing the points data

        """
        if n_states is None:
            n_states = self.n_states
        assert n_states is not None
        if states_indices is None:
            idata: dict[str, Any] = {"coords": {}, "data_vars": {}}
        else:
            idata = {
                "coords": {FC.STATE: states_indices},
                "data_vars": {},
            }

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

    def add_to_chunk_store(
        self,
        name: str,
        data: Any,
        dims: tuple[Any, ...],
        mdata: MData,
        tdata: TData | None = None,
        copy: bool = True,
        subset: Any = None,
    ) -> None:
        """
        Add data to the chunk store

        Parameters
        ----------
        name
            The data name
        data
            The data
        dims
            The data dimensions
        mdata
            The mdata object
        tdata
            The tdata object
        copy
            Flag for copying incoming data
        subset
            data corresponds to this subset of the already
            stored data, if given.

        """
        assert mdata.chunki_states is not None, (
            f"{self.name}: mdata.chunki_states is None, cannot add to chunk store"
        )
        assert mdata.chunki_points is not None, (
            f"{self.name}: mdata.chunki_points is None, cannot add to chunk store"
        )

        key = (mdata.chunki_states, mdata.chunki_points)
        if key not in self.chunk_store:
            assert mdata.n_states is not None
            n_states = int(mdata.n_states)
            n_targets = int(tdata.n_targets if tdata is not None else 0)
            self.chunk_store[key] = Dict(
                {
                    "i0": mdata.states_i0(counter=True),
                    "t0": tdata.targets_i0() if tdata is not None else 0,
                    "n_states": n_states,
                    "n_targets": n_targets,
                    "states_index": mdata[FC.STATE].copy(),
                    "dims": {"states_index": (FC.STATE,)},
                },
                _name=f"chunk_store_{key[0]}_{key[1]}",
            )

        if subset is None:
            self.chunk_store[key][name] = data.copy() if copy else data
            self.chunk_store[key]["dims"][name] = dims
        else:
            assert name in self.chunk_store[key], (
                f"{self.name}: Attempt to add subset of data '{name}' to chunk store, but full data not found for key {key}"
            )
            assert dims == self.chunk_store[key]["dims"][name], (
                f"{self.name}: Dims mismatch when adding subset of data '{name}' to chunk store, expecting {self.chunk_store[key]['dims'][name]}, got {dims}"
            )
            data0 = self.chunk_store[key][name]
            if isinstance(data0, dict):
                for k, d in data.items():
                    data0[k][subset] = d.copy() if copy else d
            else:
                self.chunk_store[key][name][subset] = data.copy() if copy else data

    def get_from_chunk_store(
        self,
        name: str,
        mdata: MData,
        prev_s: int = 0,
        prev_t: int = 0,
        ret_inds: bool = False,
        error: bool = True,
    ) -> Any:
        """
        Get data from the chunk store

        Parameters
        ----------
        name
            The data name
        mdata
            The mdata object
        tdata
            The tdata object
        prev_s
            How many states chunks backward
        prev_t
            How many points chunks backward
        ret_inds
            Also return chunk index data (i0, n_states, t0, n_targets)
        error
            Flag for raising KeyError if data not found

        Returns
        -------
        data
            The data
        inds
            The chunk index data (i0, n_states, t0, n_targets)

        """
        assert mdata.chunki_states is not None, (
            f"{self.name}: mdata.chunki_states is None, cannot add to chunk store"
        )
        assert mdata.chunki_points is not None, (
            f"{self.name}: mdata.chunki_points is None, cannot add to chunk store"
        )
        key = (mdata.chunki_states - prev_s, mdata.chunki_points - prev_t)
        try:
            chunk_data = self.chunk_store[key]
        except KeyError as e:
            if error:
                raise e
            else:
                return (None, (None, None, None, None)) if ret_inds else None

        chunk_states = chunk_data["states_index"]
        n_states = len(chunk_states)
        if (
            prev_s != 0
            or prev_t != 0
            or name not in chunk_data["dims"]
            or FC.STATE not in chunk_data["dims"][name]
            or (n_states == mdata.n_states and np.all(chunk_states == mdata[FC.STATE]))
        ):
            try:
                data = chunk_data[name]
            except KeyError as e:
                if error:
                    raise e
                else:
                    data = None

        # combine data from multiple chunks, in case of states subset selection:
        else:
            data = None
            for (_, ipoints), d in self.chunk_store.items():
                if ipoints == key[1] and name in d:
                    _, j0, j1 = np.intersect1d(
                        d["states_index"], mdata[FC.STATE], return_indices=True
                    )
                    if len(j0) == 0 or len(j1) == 0:
                        assert len(j0) == 0 and len(j1) == 0
                    elif isinstance(d[name], dict):
                        if data is None:
                            data = {}
                        else:
                            assert isinstance(data, dict)
                        for k in d[name]:
                            if k not in data:
                                data[k] = np.full(
                                    (mdata.n_states,) + d[name][k].shape[1:], np.nan
                                )
                            data[k][j1] = d[name][k][j0]
                    else:
                        if data is None:
                            data = np.full(
                                (mdata.n_states,) + d[name].shape[1:], np.nan
                            )
                        assert isinstance(data, np.ndarray)
                        data[j1] = d[name][j0]

            if data is None and error:
                raise KeyError(
                    f"{self.name}: Data '{name}' not found in chunk store for key {key}"
                )

        if ret_inds:
            inds = (
                chunk_data["i0"],
                chunk_data["n_states"],
                chunk_data["t0"],
                chunk_data["n_targets"],
            )
            return data, inds
        else:
            return data

    def reset_chunk_store(self, new_chunk_store: Any = None) -> Dict:
        """
        Resets the chunk store

        Parameters
        ----------
        new_chunk_store
            The new chunk store

        Returns
        -------
        chunk_store
            The chunk store before resetting

        """
        chunk_store = self.chunk_store
        if new_chunk_store is None:
            self.__chunk_store = Dict(_name="chunk_store")
        elif isinstance(new_chunk_store, Dict):
            self.__chunk_store = new_chunk_store
        else:
            self.__chunk_store = Dict(_name="chunk_store")
            self.__chunk_store.update(new_chunk_store)
        return chunk_store

    def block_convergence(self, **kwargs: Any) -> None:
        """
        Switch on convergence block during iterative run

        Parameters
        ----------
        kwargs
            Parameters for add_to_chunk_store()

        """
        self.add_to_chunk_store(
            name=FC.BLOCK_CONVERGENCE, data=True, dims=(), copy=False, **kwargs
        )

    def eval_conv_block(self) -> bool:
        """
        Evaluate convergence block, removing blocks on the fly

        Returns
        -------
        blocked
            True if convergence is currently blocked

        """
        blocked = False
        for c in self.__chunk_store.values():
            blocked = c.pop(FC.BLOCK_CONVERGENCE, False) or blocked
        return blocked

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        isel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Set this model status to running and move all large data to stash.

        The stashed data is returned by ``unset_running`` after the calculations.

        Parameters
        ----------
        algo
            The calculation algorithm.
        data_stash
            Large-data stash. This function adds data here when provided.
            Keys are model names and values are dictionaries of large model data.
        sel
            The subset selection dictionary.
        isel
            The index subset selection dictionary.
        verbosity
            The verbosity level; ``0`` is silent.

        """
        assert algo is self

        super().set_running(algo, data_stash, sel, isel, verbosity=verbosity)

        if data_stash is not None:
            data_stash[self.name].update(
                dict(
                    mbook=self.__mbook,
                    dbook=self.__dbook,
                    loaded_data=self.__loaded_data,
                )
            )
        del self.__mbook, self.__dbook, self.__loaded_data

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        isel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Set this model status to not running and recover large data from stash.

        Parameters
        ----------
        algo
            The calculation algorithm.
        data_stash
            Reconstruct model data from this stash when provided.
            Keys are model names and values are dictionaries of large model data.
        sel
            The subset selection dictionary.
        isel
            The index subset selection dictionary.
        verbosity
            The verbosity level; ``0`` is silent.

        """
        assert algo is self

        super().unset_running(algo, data_stash, sel, isel, verbosity)

        if data_stash is not None:
            data = data_stash[self.name]
            self.__mbook = data.pop("mbook")
            self.__dbook = data.pop("dbook")
            self.__loaded_data = data.pop("loaded_data")
        else:
            self.reset_chunk_store()

    def _launch_parallel_farm_calc(self, *args: Any, **kwargs: Any) -> xr.Dataset:
        """
        Run the farm calculation in parallel.

        Parameters
        ----------
        args
            Additional parameters for running.
        kwargs
            Additional keyword parameters for running.

        Returns
        -------
        farm_results
            The farm results. The calculated variables have dimensions
            ``(state, turbine)``.

        """
        return launch_parallel_calc(self, *args, **kwargs)

    def calc_farm(
        self,
        *args: Any,
        clear_mem: bool = False,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Calculate farm data.

        Parameters
        ----------
        args
            Positional parameters.
        clear_mem
            Clear in-memory data after starting the run.
        kwargs
            Keyword parameters.

        Returns
        -------
        farm_results
            The farm results. The calculated variables have dimensions
            ``(state, turbine)``.

        """
        if self.running:
            raise ValueError(
                f"Algorithm '{self.name}': Cannot call calc_farm while running"
            )

        # set to running:
        data_stash: dict[str, Any] | None = {} if not clear_mem else None
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
            **kwargs,
        )

        # reset to not running:
        for m in mdls:
            m.unset_running(
                self, data_stash, sel=None, isel=None, verbosity=self.verbosity - 2
            )

        return farm_results

    def _launch_parallel_points_calc(self, *args: Any, **kwargs: Any) -> xr.Dataset:
        """
        Runs the main points calculation in parallel

        Parameters
        ----------
        args
            Additional parameters for running
        kwargs
            Additional parameters for running

        Returns
        -------
        point_results
            The point results. The calculated variables have
            dimensions (state, point)

        """
        return launch_parallel_calc(self, *args, **kwargs)

    def calc_points(
        self,
        *args: Any,
        sel: dict[str, Any] | None = None,
        isel: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Calculate points data.

        Parameters
        ----------
        args
            Parameters
        sel
            The subset selection dictionary
        isel
            The index subset selection dictionary
        kwargs
            Keyword parameters

        Returns
        -------
        point_results
            The point results. The calculated variables have
            dimensions (state, point)

        """
        if self.running:
            raise ValueError(
                f"Algorithm '{self.name}': Cannot call calc_points while running"
            )

        # set to running:
        data_stash: dict[str, Any] = {}
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
        self.reset_chunk_store(chunk_store)

        # reset to not running:
        self.unset_running(
            self, data_stash, sel=sel, isel=isel, verbosity=self.verbosity - 2
        )

        return point_results

    def finalize(self, clear_mem: bool = False) -> None:
        """
        Finalizes the algorithm.

        Parameters
        ----------
        clear_mem
            Clear idata memory

        """
        if self.running:
            raise ValueError(f"Algorithm '{self.name}': Cannot finalize while running")
        super().finalize(self, self.verbosity - 1)
        if clear_mem:
            pass
            # self.reset_chunk_store()

    @classmethod
    def new(cls, algo_type: str, *args: Any, **kwargs: Any) -> Algorithm:
        """
        Run-time algorithm factory.

        Parameters
        ----------
        algo_type
            The selected derived class name
        args
            Additional parameters for the constructor
        kwargs
            Additional parameters for the constructor

        """
        return cast(Algorithm, new_instance(cls, algo_type, *args, **kwargs))
