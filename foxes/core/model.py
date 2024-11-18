import numpy as np
from abc import ABC
from itertools import count

from foxes.config import config
import foxes.constants as FC


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

        self.__initialized = False
        self.__running = False

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
        if self.initialized:
            raise ValueError(
                f"Model '{self.name}': Cannot call load_data after initialization"
            )
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
        if self.running:
            raise ValueError(f"Model '{self.name}': Cannot initialize while running")
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

    @property
    def running(self):
        """
        Flag for currently running models

        Returns
        -------
        flag: bool
            True if currently running

        """
        return self.__running

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
        if self.running:
            raise ValueError(
                f"Model '{self.name}': Cannot call set_running while running"
            )
        for m in self.sub_models():
            if not m.running:
                m.set_running(algo, data_stash, sel, isel, verbosity=verbosity)

        if verbosity > 0:
            print(f"Model '{self.name}': running")
        if self.name not in data_stash:
            data_stash[self.name] = {}

        self.__running = True

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
        if not self.running:
            raise ValueError(
                f"Model '{self.name}': Cannot call unset_running when not running"
            )
        for m in self.sub_models():
            if m.running:
                m.unset_running(algo, data_stash, sel, isel, verbosity=verbosity)

        if verbosity > 0:
            print(f"Model '{self.name}': not running")
        self.__running = False

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
        if self.running:
            raise ValueError(f"Model '{self.name}': Cannot finalize while running")
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
        selection=None,
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
        selection: numpy.ndarray, optional
            Apply this selection to the result,
            state-turbine, state-target, or state-target-tpoint

        """

        def _geta(a):
            sources = [s for s in [mdata, fdata, tdata, algo, self] if s is not None]
            for s in sources:
                try:
                    if a == "states_i0":
                        out = s.states_i0(counter=True)
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
            if downwind_index is not None:
                raise ValueError(
                    f"Target '{target}' is incompatible with downwind_index (here {downwind_index})"
                )
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

        def _match_shape(a):
            out = np.asarray(a)
            if len(out.shape) < len(shp):
                for i, s in enumerate(shp):
                    if i >= len(out.shape):
                        out = out[..., None]
                    elif a.shape[i] not in (1, s):
                        raise ValueError(
                            f"Shape mismatch for '{variable}': Got {out.shape}, expecting {shp}"
                        )
            elif len(out.shape) > len(shp):
                raise ValueError(
                    f"Shape mismatch for '{variable}': Got {out.shape}, expecting {shp}"
                )
            return out

        def _filter_dims(source):
            a = source[variable]
            a_dims = tuple(source.dims[variable])
            if downwind_index is None or FC.TURBINE not in a_dims:
                d = a_dims
            else:
                slc = tuple(
                    [downwind_index if dd == FC.TURBINE else np.s_[:] for dd in a_dims]
                )
                a = a[slc]
                d = tuple([dd for dd in a_dims if dd != FC.TURBINE])
            return a, d

        out = None
        for s in lookup:
            # lookup self:
            if s == "s" and hasattr(self, variable):
                a = getattr(self, variable)
                if a is not None:
                    out = _match_shape(a)

            # lookup mdata:
            elif s == "m" and mdata is not None and variable in mdata:
                a, d = _filter_dims(mdata)
                l = len(d)
                if l <= len(dims) and d == dims[:l]:
                    out = _match_shape(mdata[variable])

            # lookup fdata:
            elif (
                s == "f"
                and fdata is not None
                and variable in fdata
                and tuple(fdata.dims[variable]) == (FC.STATE, FC.TURBINE)
            ):
                if target == FC.STATE_TURBINE:
                    out = fdata[variable]
                elif downwind_index is not None:
                    out = _match_shape(fdata[variable][:, downwind_index])

            # lookup pdata:
            elif (
                s == "t"
                and target != FC.STATE_TURBINE
                and tdata is not None
                and variable in tdata
            ):
                a, d = _filter_dims(tdata)
                l = len(d)
                if l <= len(dims) and d == dims[:l]:
                    out = _match_shape(tdata[variable])

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
                out = _match_shape(
                    algo.wake_frame.get_wake_modelling_data(
                        algo,
                        variable,
                        downwind_index,
                        fdata,
                        tdata=tdata,
                        target=target,
                    )
                )

            if out is not None:
                break

        # check for None:
        if out is None:
            if not accept_none:
                raise ValueError(
                    f"Model '{self.name}': Variable '{variable}' is requested but not found."
                )
            return out

        # data from other chunks, only with iterations:
        if (
            target in [FC.STATE_TARGET, FC.STATE_TARGET_TPOINT]
            and fdata is not None
            and variable in fdata
            and tdata is not None
            and FC.STATES_SEL in tdata
        ):
            if out.shape != shp:
                # upcast to dims:
                tmp = np.zeros(shp, dtype=out.dtype)
                tmp[:] = out
                out = tmp
                del tmp
            else:
                out = out.copy()
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

            from foxes.algorithms.sequential import Sequential

            if isinstance(algo, Sequential):
                i0 = algo.states.counter
            else:
                i0 = _geta("states_i0")
            sts = tdata[FC.STATES_SEL]
            if target == FC.STATE_TARGET and tdata.n_tpoints != 1:
                # find the mean index and round it to nearest integer:
                sts = tdata.tpoint_mean(FC.STATES_SEL)[:, :, None]
                sts = (sts + 0.5).astype(config.dtype_int)
            sel = sts < i0
            if np.any(sel):
                if not hasattr(algo, "farm_results_downwind"):
                    raise KeyError(
                        f"Model '{self.name}': Iteration data found for variable '{variable}', requiring iterative algorithm"
                    )
                prev_fres = getattr(algo, "farm_results_downwind")
                if prev_fres is not None:
                    prev_data = prev_fres[variable].to_numpy()[sts[sel], downwind_index]
                    if target == FC.STATE_TARGET:
                        out[sel[:, :, 0]] = prev_data
                    else:
                        out[sel] = prev_data
                    del prev_data
                del prev_fres
            if np.any(~sel):
                sts = sts[~sel] - i0
                sel_data = fdata[variable][sts, downwind_index]
                if target == FC.STATE_TARGET:
                    out[~sel[:, :, 0]] = sel_data
                else:
                    out[~sel] = sel_data
                del sel_data
            del sel, sts

        # check for nan:
        if not accept_nan:
            try:
                if np.all(np.isnan(np.atleast_1d(out))):
                    raise ValueError(
                        f"Model '{self.name}': Requested variable '{variable}' contains NaN values."
                    )
            except TypeError:
                pass

        # apply selection:
        if selection is not None:

            def _upcast_sel(sel_shape):
                chp = []
                for i, s in enumerate(out.shape):
                    if i < len(sel_shape) and sel_shape[i] > 1:
                        if sel_shape[i] != shp[i]:
                            raise ValueError(
                                f"Incompatible selection shape {sel_shape} for output shape {shp[i]}"
                            )
                        chp.append(shp[i])
                    else:
                        chp.append(s)
                chp = tuple(chp)
                eshp = list(shp[len(sel_shape) :])
                if chp != out.shape:
                    nout = np.zeros(chp, dtype=out.dtype)
                    nout[:] = out
                    return nout, eshp
                return out, eshp

            if isinstance(selection, np.ndarray) and selection.dtype == bool:
                if len(selection.shape) > len(out.shape):
                    raise ValueError(
                        f"Expecting selection of shape {out.shape}, got {selection.shape}"
                    )
                out, eshp = _upcast_sel(selection.shape)
            elif isinstance(selection, (tuple, list)):
                if len(selection) > len(out.shape):
                    raise ValueError(
                        f"Selection is tuple/list of length {len(selection)}, expecting <= {len(out.shape)} "
                    )
                out, eshp = _upcast_sel(shp[: len(selection)])
            else:
                raise TypeError(
                    f"Expecting selection of type np.ndarray (bool), or tuple, or list. Got {type(selection).__name__}"
                )
            out = out[selection]
            shp = tuple([len(out)] + list(eshp))

        # apply upcast:
        if upcast and out.shape != shp:
            tmp = np.zeros(shp, dtype=out.dtype)
            tmp[:] = out
            out = tmp
            del tmp

        return out
