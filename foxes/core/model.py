import numpy as np
from abc import ABC
from itertools import count
from copy import deepcopy

import foxes.constants as FC
from .data import Data


class Model(ABC):
    """
    Base class for all models.

    Attributes
    ----------
    name: str
        The model name

    :group: core

    """

    _ids = {}

    def __init__(self):
        """
        Constructor.
        """
        t = type(self).__name__
        if t not in self._ids:
            self._ids[t] = count(0)
        self._id = next(self._ids[t])

        self.name = f"{type(self).__name__}"
        if self._id > 0:
            self.name += f"_instance{self._id}"

        self._store = {}
        self.__initialized = False

    def __repr__(self):
        return f"{type(self).__name__}()"

    @property
    def model_id(self):
        """
        Unique id based on the model type.

        Returns
        -------
        int
            Unique id of the model object

        """
        return self._id

    def var(self, v):
        """
        Creates a model specific variable name.

        Parameters
        ----------
        v: str
            The variable name

        Returns
        -------
        str
            Model specific variable name

        """
        return f"{self.name}_{v}"

    @property
    def initialized(self):
        """
        Initialization flag.

        Returns
        -------
        bool :
            True if the model has been initialized.

        """
        return self.__initialized

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return []

    def load_data(self, algo, verbosity=0):
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        idata: dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        return {"coords": {}, "data_vars": {}}

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        if not self.initialized:
            pr = False
            for m in self.sub_models():
                if not m.initialized:
                    if verbosity > 1 and not pr:
                        print(f">> {self.name}: Starting sub-model initialization >> ")
                        pr = True
                    m.initialize(algo, verbosity)
            if pr:
                print(f"<< {self.name}: Finished sub-model initialization << ")

            if verbosity > 0:
                print(f"Initializing model '{self.name}'")
            algo.store_model_data(self, self.load_data(algo, verbosity), force)

            self.__initialized = True

    def finalize(self, algo, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        if self.initialized:
            pr = False
            for m in self.sub_models():
                if verbosity > 1 and not pr:
                    print(f">> {self.name}: Starting sub-model finalization >> ")
                    pr = True
                m.finalize(algo, verbosity)
            if pr:
                print(f"<< {self.name}: Finished sub-model finalization << ")

            if verbosity > 0:
                print(f"Finalizing model '{self.name}'")
            algo.del_model_data(self)

            self._store = {}
            self.__initialized = False

    def get_data(
        self,
        variable,
        target,
        lookup="smfp",
        mdata=None,
        fdata=None,
        tdata=None,
        downwind_index=None,
        accept_none=False,
        accept_nan=True,
        algo=None,
        upcast=False,
    ):
        """
        Getter for a data entry in the model object
        or provided data sources

        Parameters
        ----------
        variable: str
            The variable, serves as data key
        target: str, optional
            The dimensions identifier for the output,
            FC.STATE_TURBINE, FC.STATE_TARGET or
            FC.STATE_TARGET_TPOINT
        lookup: str
            The order of data sources. Combination of:
            's' for self,
            'm' for mdata,
            'f' for fdata,
            't' for tdata,
            'w' for wake modelling data
        mdata: foxes.core.Data, optional
            The model data
        fdata: foxes.core.Data, optional
            The farm data
        tdata: foxes.core.Data, optional
            The target point data
        downwind_index: int, optional
            The index in the downwind order
        data_prio: bool
            First search the data source, then the object
        accept_none: bool
            Do not throw an error if data entry is None
        accept_nan: bool
            Do not throw an error if data entry is np.nan
        algo: foxes.core.Algorithm, optional
            The algorithm, needed for data from previous iteration
        upcast: bool
            Flag for ensuring targets dimension,
            otherwise dimension 1 is entered

        """

        def _geta(a):
            sources = [s for s in [mdata, fdata, tdata, algo, self] if s is not None]
            for s in sources:
                try:
                    if a == "states_i0":
                        out = s.states_i0(counter=True, algo=algo)
                        if out is not None:
                            return out
                    else:
                        out = getattr(s, a)
                        if out is not None:
                            return out
                except AttributeError:
                    pass
            raise KeyError(
                f"Model '{self.name}': Failed to determine '{a}'. Maybe add to arguments of get_data: mdata, fdata, tdata, algo?"
            )

        n_states = _geta("n_states")
        if target == FC.STATE_TURBINE:
            n_turbines = _geta("n_turbines")
            dims = (FC.STATE, FC.TURBINE)
            shp = (n_states, n_turbines)
        elif target == FC.STATE_TARGET:
            n_targets = _geta("n_targets")
            dims = (FC.STATE, FC.TARGET)
            shp = (n_states, n_targets)
        elif target == FC.STATE_TARGET_TPOINT:
            n_targets = _geta("n_targets")
            n_tpoints = _geta("n_tpoints")
            dims = (FC.STATE, FC.TARGET, FC.TPOINT)
            shp = (n_states, n_targets, n_tpoints)
        else:
            raise KeyError(
                f"Model '{self.name}': Wrong parameter 'target = {target}'. Choices: {FC.STATE_TURBINE}, {FC.STATE_TARGET}, {FC.STATE_TARGET_TPOINT}"
            )

        out = None
        out_dims = None
        for s in lookup:
            # lookup self:
            if s == "s" and hasattr(self, variable):
                a = getattr(self, variable)
                if a is not None:
                    if not upcast:
                        out = a
                        out_dims = None
                    elif target == FC.STATE_TURBINE:
                        out = np.full((n_states, n_turbines), np.nan, dtype=FC.DTYPE)
                        out[:] = a
                        out_dims = (FC.STATE, FC.TURBINE)
                    elif target == FC.STATE_TARGET:
                        out = np.full((n_states, n_targets), np.nan, dtype=FC.DTYPE)
                        out[:] = a
                        out_dims = (FC.STATE, FC.TARGET)
                    elif target == FC.STATE_TARGET_TPOINT:
                        out = np.full(
                            (n_states, n_targets, n_tpoints), np.nan, dtype=FC.DTYPE
                        )
                        out[:] = a
                        out_dims = (FC.STATE, FC.TARGET, FC.TPOINT)
                    else:
                        raise NotImplementedError

            # lookup mdata:
            elif (
                s == "m"
                and mdata is not None
                and variable in mdata
                and tuple(mdata.dims[variable]) == dims
            ):
                out = mdata[variable]
                out_dims = dims

            # lookup fdata:
            elif (
                s == "f"
                and fdata is not None
                and variable in fdata
                and tuple(fdata.dims[variable]) == (FC.STATE, FC.TURBINE)
            ):
                out = fdata[variable]
                out_dims = (FC.STATE, FC.TURBINE)

            # lookup pdata:
            elif (
                s == "t"
                and tdata is not None
                and variable in tdata
                and tuple(tdata.dims[variable]) == (FC.STATE, FC.TARGET, FC.TPOINT)
            ):
                out = tdata[variable]
                out_dims = (FC.STATE, FC.TARGET, FC.TPOINT)

            # lookup wake modelling data:
            elif (
                s == "w"
                and fdata is not None
                and tdata is not None
                and variable in fdata
                and tuple(fdata.dims[variable]) == (FC.STATE, FC.TURBINE)
                and downwind_index is not None
                and algo is not None
            ):
                out, out_dims = algo.wake_frame.get_wake_modelling_data(
                    algo,
                    variable,
                    downwind_index,
                    fdata,
                    tdata=tdata,
                    target=target,
                    upcast=upcast,
                )

            if out is not None:
                break

        # cast dimensions:
        if out_dims != dims:
            if out_dims is None:
                if upcast:
                    out0 = out
                    out = np.zeros(shp, dtype=FC.DTYPE)
                    out[:] = out0
                    out_dims = dims
                    del out0
                else:
                    out_dims = tuple([1 for _ in dims])

            elif out_dims == (FC.STATE, FC.TURBINE):
                if downwind_index is None:
                    raise KeyError(
                        f"Require downwind_index for target {target} and out dims {out_dims}"
                    )
                out0 = out[:, downwind_index, None]
                if len(dims) == 3:
                    out0 = out0[:, :, None]
                if upcast:
                    out = np.zeros(shp, dtype=FC.DTYPE)
                    out[:] = out0
                    out_dims = dims
                else:
                    out = out0
                    out_dims = (FC.STATE, 1) if len(dims) == 2 else (FC.STATE, 1, 1)
                del out0

            elif out_dims == (FC.STATE, 1):
                out0 = out
                if len(dims) == 3:
                    out0 = out0[:, :, None]
                    out_dims = (FC.STATE, 1, 1)
                if upcast:
                    out = np.zeros(shp, dtype=FC.DTYPE)
                    out[:] = out0
                    out_dims = dims
                else:
                    out = out0
                del out0

            elif out_dims == (FC.STATE, 1, 1):
                out0 = out
                if len(dims) == 2:
                    out0 = out0[:, :, 0]
                    out_dims = (FC.STATE, 1)
                if upcast:
                    out = np.zeros(shp, dtype=FC.DTYPE)
                    out[:] = out0
                    out_dims = dims
                else:
                    out = out0
                del out0

            else:
                raise NotImplementedError(
                    f"No casting implemented for target {target} and out dims {out_dims} fo upcast {upcast}"
                )

        # data from other chunks, only with iterations:
        if (
            target in [FC.STATE_TARGET, FC.STATE_TARGET_TPOINT]
            and fdata is not None
            and variable in fdata
            and tdata is not None
            and FC.STATES_SEL in tdata
        ):
            if out_dims != dims:
                raise ValueError(
                    f"Model '{self.name}': Iteration data found for variable '{variable}', but missing upcast: out_dims = {out_dims}, expecting {dims}"
                )
            if downwind_index is None:
                raise KeyError(
                    f"Model '{self.name}': Require downwind_index for obtaining results from previous iteration"
                )
            if tdata[FC.STATE_SOURCE_ORDERI] != downwind_index:
                raise ValueError(
                    f"Model '{self.name}': Expecting downwind_index {tdata[FC.STATE_SOURCE_ORDERI]}, got {downwind_index}"
                )
            if algo is None:
                raise ValueError(
                    f"Model '{self.name}': Iteration data found for variable '{variable}', requiring algo"
                )

            i0 = _geta("states_i0")
            sts = tdata[FC.STATES_SEL]
            if target == FC.STATE_TARGET and tdata.n_tpoints != 1:
                # find the mean index and round it to nearest integer:
                sts = tdata.tpoint_mean(FC.STATES_SEL)[:, :, None]
                sts = (sts + 0.5).astype(FC.ITYPE)
            sel = sts < i0
            if np.any(sel):
                if not hasattr(algo, "prev_farm_results"):
                    raise KeyError(
                        f"Model '{self.name}': Iteration data found for variable '{variable}', requiring iterative algorithm"
                    )
                prev_fres = getattr(algo, "prev_farm_results")
                if prev_fres is not None:
                    prev_data = prev_fres[variable].to_numpy()[sts[sel], downwind_index]
                    if target == FC.STATE_TARGET:
                        out[sel[:, :, 0]] = prev_data
                    else:
                        out[sel] = prev_data
                    del prev_fres, prev_data

        # check for None:
        if not accept_none and out is None:
            raise ValueError(
                f"Model '{self.name}': Variable '{variable}' is requested but not found."
            )

        # check for nan:
        elif not accept_nan:
            try:
                if np.all(np.isnan(np.atleast_1d(out))):
                    raise ValueError(
                        f"Model '{self.name}': Requested variable '{variable}' contains NaN values."
                    )
            except TypeError:
                pass

        return out

    def data_to_store(self, name, algo, data):
        """
        Adds data from mdata to the local store, intended
        for iterative runs.

        Parameters
        ----------
        name: str
            The data name
        algo: foxes.core.Algorithm
            The algorithm
        data: foxes.utils.Data
            The mdata, fdata or pdata object

        """
        i0 = data.states_i0(counter=True, algo=algo)
        if i0 not in self._store:
            self._store[i0] = Data(
                data={}, dims={}, loop_dims=data.loop_dims, name=f"{self.name}_{i0}"
            )

        self._store[i0][name] = deepcopy(data[name])
        self._store[i0].dims[name] = (
            deepcopy(data.dims[name]) if name in data.dims else None
        )

    def from_data_or_store(self, name, algo, data, ret_dims=False, safe=False):
        """
        Get data from mdata or local store

        Parameters
        ----------
        name: str
            The data name
        algo: foxes.core.Algorithm
            The algorithm
        data: foxes.utils.Data
            The mdata, fdata or pdata object
        ret_dims: bool
            Return dimensions
        safe: bool
            Return None instead of error if
            not found

        Returns
        -------
        data: numpy.ndarray
            The data
        dims: tuple of dims, optional
            The data dimensions

        """
        if name in data:
            return (data[name], data.dims[name]) if ret_dims else data[name]

        i0 = data.states_i0(counter=True, algo=algo)
        if not safe or (i0 in self._store and name in self._store[i0]):
            if ret_dims:
                return self._store[i0][name], self._store[i0].dims[name]
            else:
                return self._store[i0][name]
        else:
            return (None, None) if ret_dims else None
