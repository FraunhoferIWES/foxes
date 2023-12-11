import numpy as np
from abc import ABCMeta
from itertools import count

import foxes.constants as FC
from .data import Data


class Model(metaclass=ABCMeta):
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

        ext = "" if self._id == 0 else f"{self._id}"
        self.name = f"{type(self).__name__}{ext}"

        self._store = {}
        self.__initialized = False

    def __repr__(self):
        t = type(self).__name__
        return f"{self.name} ({t})"

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
        pdata=None,
        states_source_turbine=None,
        upcast=False,
        accept_none=False,
        accept_nan=True,
        algo=None,
    ):
        """
        Getter for a data entry in the model object
        or provided data sources

        Parameters
        ----------
        variable: str
            The variable, serves as data key
        target: str, optional
            The dimensions identifier for the output, e.g
            FC.STATE_TURBINE, FC.STATE_POINT
        lookup: str
            The order of data sources. Combination of:
            's' for self,
            'm' for mdata,
            'f' for fdata,
            'p' for pdata,
            'w' for wake modelling data
        mdata: foxes.core.Data, optional
            The model data
        fdata: foxes.core.Data, optional
            The farm data
        pdata: foxes.core.Data, optional
            The evaluation point data
        states_source_turbine: numpy.ndarray, optional
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        upcast: bool, optional
            Upcast array to dims if data is scalar
        data_prio: bool
            First search the data source, then the object
        accept_none: bool
            Do not throw an error if data entry is None
        accept_nan: bool
            Do not throw an error if data entry is np.nan
        algo: foxes.core.Algorithm, optional
            The algorithm, needed for data from previous iteration

        """

        def _geta(a):
            sources = [s for s in [mdata, fdata, pdata, algo, self] if s is not None]
            for s in sources:
                if a == "states_i0":
                    out = s.states_i0(counter=True, algo=algo)
                    if out is not None:
                        return out
                else:
                    try:
                        out = getattr(s, a)
                        if out is not None:
                            return out
                    except AttributeError:
                        pass
            raise KeyError(
                f"Model '{self.name}': Failed to determine '{a}'. Maybe add to arguments of get_data: mdata, fdata, pdata, algo?"
            )

        n_states = _geta("n_states")
        if target == FC.STATE_TURBINE:
            n_turbines = _geta("n_turbines")
            dims = (FC.STATE, FC.TURBINE)
        elif target == FC.STATE_POINT:
            n_points = _geta("n_points")
            dims = (FC.STATE, FC.POINT)
        else:
            raise KeyError(
                f"Model '{self.name}': Wrong parameter 'target = {target}'. Choices: {FC.STATE_TURBINE}, {FC.STATE_POINT}"
            )

        out = None
        for s in lookup:
            # lookup self:
            if s == "s" and hasattr(self, variable):
                a = getattr(self, variable)

                if a is not None and upcast:
                    if target == FC.STATE_TURBINE:
                        out = np.full((n_states, n_turbines), np.nan, dtype=FC.DTYPE)
                        out[:] = a
                    elif target == FC.STATE_POINT:
                        out = np.full((n_states, n_points), np.nan, dtype=FC.DTYPE)
                        out[:] = a
                    else:
                        raise KeyError(
                            f"Model '{self.name}': Wrong parameter 'target = {target}' for 'upcast = True' in get_data. Choose: FC.STATE_TURBINE, FC.STATE_POINT"
                        )

                else:
                    out = a

            # lookup mdata:
            elif (
                s == "m"
                and mdata is not None
                and variable in mdata
                and len(mdata.dims[variable]) > 1
                and tuple(mdata.dims[variable][:2]) == dims
            ):
                out = mdata[variable]

            # lookup fdata:
            elif (
                s == "f"
                and fdata is not None
                and variable in fdata
                and len(fdata.dims[variable]) > 1
                and tuple(fdata.dims[variable][:2]) == (FC.STATE, FC.TURBINE)
            ):
                # direct fdata:
                if target == FC.STATE_TURBINE:
                    out = fdata[variable]

                # translate state-turbine to state-point data:
                elif target == FC.STATE_POINT and states_source_turbine is not None:
                    # from fdata, uniform for points:
                    st_sel = (np.arange(n_states), states_source_turbine)
                    out = np.zeros((n_states, n_points), dtype=FC.DTYPE)
                    out[:] = fdata[variable][st_sel][:, None]

                    # from previous iteration, if requested:
                    if pdata is not None and FC.STATES_SEL in pdata:
                        if not np.all(
                            states_source_turbine == pdata[FC.STATE_SOURCE_TURBINE]
                        ):
                            raise ValueError(
                                f"Model '{self.name}': Mismatch of 'states_source_turbine'. Expected {list(pdata[FC.STATE_SOURCE_TURBINE])}, got {list(states_source_turbine)}"
                            )

                        i0 = _geta("states_i0")
                        sp = pdata[FC.STATES_SEL]
                        sel = sp < i0
                        if np.any(sel):
                            if algo is None or not hasattr(algo, "prev_farm_results"):
                                raise KeyError(
                                    f"Model '{self.name}': Argument algo is either not given, or not an iterative algorithm"
                                )

                            prev_fdata = getattr(algo, "prev_farm_results")
                            if prev_fdata is None:
                                out[sel] = 0
                            else:
                                st = np.zeros_like(sp)
                                st[:] = states_source_turbine[:, None]
                                out[sel] = prev_fdata[variable].to_numpy()[
                                    sp[sel], st[sel]
                                ]

            # lookup pdata:
            elif (
                s == "p"
                and pdata is not None
                and variable in pdata
                and len(pdata.dims[variable]) > 1
                and tuple(pdata.dims[variable][:2]) == dims
            ):
                out = pdata[variable]

            # lookup wake modelling data:
            elif (
                s == "w"
                and target == FC.STATE_POINT
                and fdata is not None
                and pdata is not None
                and variable in fdata
                and len(fdata.dims[variable]) > 1
                and tuple(fdata.dims[variable][:2]) == (FC.STATE, FC.TURBINE)
                and states_source_turbine is not None
                and algo is not None
            ):
                out = algo.wake_frame.get_wake_modelling_data(
                    algo, variable, states_source_turbine, fdata, pdata
                )

            if out is not None:
                break

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

        self._store[i0][name] = data[name]
        self._store[i0].dims[name] = data.dims[name] if name in data.dims else None

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

    '''
    @classmethod
    def reduce_states(cls, sel_states, objs):
        """
        Modifies the given objects by selecting a
        subset of states.

        Parameters
        ----------
        sel_states: list of int
            The states selection
        objs: list of foxes.core.Data
            The objects, e.g. [mdata, fdata, pdata]

        Returns
        -------
        mobjs: list of foxes.core.Data
            The modified objects with reduced
            states dimension

        """
        out = []
        for o in objs:
            data = {
                v: d[sel_states] if o.dims[v][0] == FC.STATE else d
                for v, d in o.items()
            }
            out.append(Data(data, o.dims, loop_dims=o.loop_dims, name=o.name))

        return out
        '''
