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

        self.__initialized = False

    def __repr__(self):
        t = type(self).__name__
        return f"{self.name} ({t})"
    
    def keep(self, algo):
        """
        Add model and all sub models to
        the keep_models list

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm

        """
        algo.keep_models.add(self.name)

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

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        This includes loading all required data from files. The model
        should return all array type data as part of the idata return
        dictionary (and not store it under self, for memory reasons). This
        data will then be chunked and provided as part of the mdata object
        during calculations.

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
        if self.initialized:
            raise ValueError(
                f"Model '{self.name}': initialize called for already initialized object"
            )
        self.__initialized = True
        return {"coords": {}, "data_vars": {}}

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
        if not self.initialized:
            raise ValueError(
                f"Model '{self.name}': Finalization called for uninitialized object"
            )
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
            'p' for pdata
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
            Do not throw an error if data entry is None or np.nan
        algo: foxes.core.Algorithm, optional
            The algorithm, needed for data from previous iteration

        """

        def _geta(a):
            sources = [s for s in [mdata, fdata, pdata, algo, self] if s is not None]
            for s in sources:
                try:
                    out = getattr(s, a)
                    if out is not None:
                        return out
                except AttributeError:
                    pass
            raise KeyError(f"Model '{self.name}': Failed to determine '{a}'. Maybe add to arguments of get_data: mdata, fdata, pdata, algo?")

        n_states = _geta("n_states")
        if target == FC.STATE_TURBINE:
            n_turbines = _geta("n_turbines")
            dims = (FC.STATE, FC.TURBINE)
        elif target == FC.STATE_POINT:
            n_points = _geta("n_points")
            dims = (FC.STATE, FC.POINT)
        else:
            raise KeyError(f"Model '{self.name}': Wrong parameter 'target = {target}'. Choices: {FC.STATE_TURBINE}, {FC.STATE_POINT}")
    
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
                        raise KeyError(f"Model '{self.name}': Wrong parameter 'target = {target}' for 'upcast = True' in get_data. Choose: FC.STATE_TURBINE, FC.STATE_POINT")
                
                else:
                    out = a

            # lookup mdata:
            elif (
                s == "m" and mdata is not None and variable in mdata
                and len(mdata.dims[variable]) > 1
                and tuple(mdata.dims[variable][:2]) == dims
            ):
                out = mdata[variable]
            
            # lookup fdata:
            elif (
                s == "f" and fdata is not None and variable in fdata
                and len(fdata.dims[variable]) > 1
                and tuple(fdata.dims[variable][:2]) == (FC.STATE, FC.TURBINE)
            ):
                # direct fdata:
                if target == FC.STATE_TURBINE:
                    out = fdata[variable]
                
                # translate state-turbine to state-point data:
                elif (
                    target == FC.STATE_POINT
                    and states_source_turbine is not None
                ):
                    # from fdata, uniform for points:
                    st_sel = (np.arange(n_states), states_source_turbine)
                    out = np.zeros((n_states, n_points),  dtype=FC.DTYPE)
                    out[:] = fdata[variable][st_sel][:, None]

                    # from previous iteration, if requested:
                    if (
                        pdata is not None
                        and FC.STATES_SEL in pdata
                    ):
                        if not np.all(states_source_turbine == pdata[FC.STATE_SOURCE_TURBINE]):
                            raise ValueError(f"Model '{self.name}': Mismatch of 'states_source_turbine'. Expected {list(pdata[FC.STATE_SOURCE_TURBINE])}, got {list(states_source_turbine)}")

                        i0 = np.argwhere(algo.states.index() == _geta("states_i0"))[0][0]
                        sp = pdata[FC.STATES_SEL]
                        sel = (sp < i0)
                        if np.any(sel):

                            if (
                                algo is None
                                or not hasattr(algo, "prev_farm_results")
                            ):
                                raise KeyError(f"Model '{self.name}': Argument algo is either not given, or not an iterative algorithm")

                            prev_fdata = getattr(algo, "prev_farm_results")
                            if prev_fdata is None:
                                out[sel] = 0
                            else:
                                st = np.zeros_like(sp)
                                st[:] = states_source_turbine[:, None]
                                out[sel] = prev_fdata[variable].to_numpy()[sp[sel], st[sel]]
                                del st

            # lookup pdata:
            elif (
                s == "p" and pdata is not None and variable in pdata
                and len(pdata.dims[variable]) > 1
                and tuple(pdata.dims[variable][:2]) == dims
            ):
                out = pdata[variable]
            
            if out is not None:
                break

        # check for None:
        if not accept_none:
            try:
                if out is None or np.all(np.isnan(np.atleast_1d(out))):
                    raise ValueError(
                        f"Model '{self.name}': Variable '{variable}' requested but not provided."
                    )
            except TypeError:
                pass

        return out

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
            out.append(
                Data(data, o.dims, loop_dims=o.loop_dims, name=o.name)
            )
                
        return out
    