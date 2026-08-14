from __future__ import annotations

import numpy as np
from abc import ABC
from itertools import count
from typing import TYPE_CHECKING, Any, Literal, TypedDict, overload

from foxes.config import config
import foxes.constants as FC


if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class LoadedData(TypedDict):
    coords: dict[str, np.ndarray]
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]]
    extra_data: dict[str, Any]


class Model(ABC):
    """
    Base class for all models.

    Attributes
    ----------
    name
        The model name

    :group: core

    """

    _ids: dict[str, Any] = {}

    def __init__(self) -> None:
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

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @property
    def model_id(self) -> int:
        """
        Unique id based on the model type.

        Returns
        -------
        int
            Unique id of the model object

        """
        return self._id

    def var(self, v: str) -> str:
        """
        Create a model-specific variable name.

        Parameters
        ----------
        v
            The variable name.

        Returns
        -------
        vnm
            The model-specific variable name.

        """
        return f"{self.name}_{v}"

    def unvar(self, vnm: str) -> str | None:
        """
        Translate a model-specific variable name to the original variable name.

        Parameters
        ----------
        vnm
            The model-specific variable name.

        Returns
        -------
        v
            The original variable name.

        """
        lng = len(f"{self.name}_")
        return vnm[lng:] if vnm.startswith(f"{self.name}_") else None

    @property
    def initialized(self) -> bool:
        """
        Initialization flag.

        Returns
        -------
        initialized
            True if the model has been initialized.

        """
        return self.__initialized

    def sub_models(self) -> list[Model]:
        """
        Return the list of all sub-models.

        Returns
        -------
        smdls
            All sub-models.

        """
        return []

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 0,
    ) -> None:
        """
        Load and/or create all data required for model calculations.

        The function adds to loaded_data.

        Parameters
        ----------
        algo
            The calculation algorithm.
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries ``dim_name_str -> dim_array``;
            "data_vars", a dict with entries ``name_str -> (dim_tuple, data_ndarray)``;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data.
        verbosity
            The verbosity level, where 0 is silent.

        """
        if self.initialized:
            raise ValueError(
                f"Model '{self.name}': Cannot call load_data after initialization"
            )

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initialize the model.

        Parameters
        ----------
        algo
            The calculation algorithm.
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries ``dim_name_str -> dim_array``;
            "data_vars", a dict with entries ``name_str -> (dim_tuple, data_ndarray)``;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data.
        verbosity
            The verbosity level, where 0 is silent.

        Returns
        -------
        loaded_data
            The loaded data, containing the keys "coords", "data_vars", and
            "extra_data".

        """

        if self.running:
            raise ValueError(f"Model '{self.name}': Cannot initialize while running")
        if loaded_data is None:
            loaded_data = {"coords": {}, "data_vars": {}, "extra_data": {}}

        if force:
            self.__initialized = False

        if not self.initialized:
            pr = False
            for m in self.sub_models():
                if force or not m.initialized:
                    if verbosity > 1 and not pr:
                        print(f">> {self.name}: Starting sub-model initialization >> ")
                        pr = True
                    m.initialize(
                        algo=algo,
                        loaded_data=loaded_data,
                        force=force,
                        verbosity=verbosity,
                    )
            if pr:
                print(f"<< {self.name}: Finished sub-model initialization << ")

            if verbosity > 0:
                print(f"Initializing model '{self.name}'")

            self.load_data(algo, loaded_data, force=force, verbosity=verbosity)

            self.__initialized = True

        return loaded_data

    @property
    def running(self) -> bool:
        """
        Flag for currently running models

        Returns
        -------
        running
            True if currently running

        """
        return self.__running

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        isel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Set this model to the running state and move large data to the stash.

        The stashed data is restored by ``unset_running`` after the calculation
        has finished.

        Parameters
        ----------
        algo
            The calculation algorithm.
        data_stash
            The large-data stash. This function adds entries here when provided.
            Keys are model names and values are dictionaries of large model data.
        sel
            The subset selection dictionary.
        isel
            The index subset selection dictionary.
        verbosity
            The verbosity level; ``0`` is silent.

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
        if data_stash is not None and self.name not in data_stash:
            data_stash[self.name] = {}

        self.__running = True

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

    def finalize(self, algo: Algorithm, verbosity: int = 0) -> None:
        """
        Finalize the model.

        Parameters
        ----------
        algo
            The calculation algorithm.
        verbosity
            The verbosity level; ``0`` is silent.

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

            self.__initialized = False

    @overload
    def get_data(
        self,
        variable: str,
        target: str,
        lookup: str = "smfp",
        mdata: MData | None = None,
        fdata: FData | None = None,
        tdata: TData | None = None,
        downwind_index: int | None = None,
        accept_none: Literal[False] = False,
        accept_nan: bool = True,
        algo: Algorithm | None = None,
        upcast: bool = False,
        selection: np.ndarray | tuple[Any, ...] | list[Any] | None = None,
    ) -> np.ndarray: ...

    @overload
    def get_data(
        self,
        variable: str,
        target: str,
        lookup: str = "smfp",
        mdata: MData | None = None,
        fdata: FData | None = None,
        tdata: TData | None = None,
        downwind_index: int | None = None,
        accept_none: Literal[True] = True,
        accept_nan: bool = True,
        algo: Algorithm | None = None,
        upcast: bool = False,
        selection: np.ndarray | tuple[Any, ...] | list[Any] | None = None,
    ) -> np.ndarray | None: ...

    def get_data(
        self,
        variable: str,
        target: str,
        lookup: str = "smfp",
        mdata: MData | None = None,
        fdata: FData | None = None,
        tdata: TData | None = None,
        downwind_index: int | None = None,
        accept_none: bool = False,
        accept_nan: bool = True,
        algo: Algorithm | None = None,
        upcast: bool = False,
        selection: np.ndarray | tuple[Any, ...] | list[Any] | None = None,
    ) -> np.ndarray | None:
        """
        Getter for a data entry in the model object
        or provided data sources

        Parameters
        ----------
        variable
            The variable name used as the data key.
        target
            The dimensions identifier for the output: ``FC.STATE_TURBINE``,
            ``FC.STATE_TARGET``, or ``FC.STATE_TARGET_TPOINT``.
        lookup
            The order of data sources. Combination of:
            ``'s'`` for self, ``'m'`` for mdata, ``'f'`` for fdata,
            ``'t'`` for tdata, and ``'w'`` for wake-modeling data.
        mdata
            The model data.
        fdata
            The farm data.
        tdata
            The target point data.
        downwind_index
            The index in the downwind order.
        accept_none
            Do not raise an error if the data entry is ``None``.
        accept_nan
            Do not raise an error if the data entry is ``np.nan``.
        algo
            The algorithm, needed for data from previous iterations.
        upcast
            Ensure the target dimension is present; otherwise dimension 1 is
            entered.
        selection
            Apply this selection to the result, for state-turbine, state-target,
            or state-target-tpoint outputs.

        """

        def _geta(a: str) -> Any:
            sources: list[Any] = [
                s for s in [mdata, fdata, tdata, algo, self] if s is not None
            ]
            for s in sources:
                try:
                    if a == "states_i0":
                        get_states_i0 = getattr(s, "states_i0", None)
                        if callable(get_states_i0):
                            out = get_states_i0(counter=True)
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

        dims: tuple[str, ...]
        shp: tuple[int, ...]
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

        def _match_shape(a: Any) -> np.ndarray:
            out = np.asarray(a)
            if len(out.shape) < len(shp):
                for i, s in enumerate(shp):
                    if i >= len(out.shape):
                        out = out[..., None]
                    elif out.shape[i] not in (1, s):
                        raise ValueError(
                            f"Shape mismatch for '{variable}': Got {out.shape}, expecting {shp}"
                        )
            elif len(out.shape) > len(shp):
                raise ValueError(
                    f"Shape mismatch for '{variable}': Got {out.shape}, expecting {shp}"
                )
            return out

        def _filter_dims(
            source: MData | FData | TData,
        ) -> tuple[np.ndarray, tuple[str, ...]]:
            a = source[variable]
            a_dims = tuple(source.dims[variable])
            if downwind_index is None or FC.TURBINE not in a_dims:
                d: tuple[str, ...] = a_dims
            else:
                slc = tuple(
                    [downwind_index if dd == FC.TURBINE else np.s_[:] for dd in a_dims]
                )
                a = a[slc]
                d = tuple(dd for dd in a_dims if dd != FC.TURBINE)
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
                ld = len(d)
                if ld <= len(dims) and d == dims[:ld]:
                    out = _match_shape(a)

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
                ld = len(d)
                if ld <= len(dims) and d == dims[:ld]:
                    out = _match_shape(a)

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
                wake_frame = algo.wake_frame
                out = _match_shape(
                    wake_frame.get_wake_modelling_data(
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
        assert out is not None

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
                i0 = getattr(algo.states, "counter", _geta("states_i0"))
            else:
                i0 = _geta("states_i0")
            sts = tdata[FC.STATES_SEL]
            if target == FC.STATE_TARGET and tdata.n_tpoints != 1:
                # find the mean index and round it to nearest integer:
                sts = tdata.tpoint_mean(FC.STATES_SEL)[:, :, None]
                sts = (sts + 0.5).astype(config.dtype_int)
            sel = sts < i0
            if np.any(sel):
                prev_fres = algo.farm_results_downwind
                if prev_fres is not None:
                    prev_data = prev_fres[variable].to_numpy()[sts[sel], downwind_index]
                    if target == FC.STATE_TARGET:
                        out[sel[:, :, 0]] = prev_data
                    else:
                        out[sel] = prev_data
                    del prev_data
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
            selected_out = out

            def _upcast_sel(sel_shape: tuple[int, ...]) -> tuple[np.ndarray, list[int]]:
                chp: list[int] = []
                for i, s in enumerate(selected_out.shape):
                    if i < len(sel_shape) and sel_shape[i] > 1:
                        if sel_shape[i] != shp[i]:
                            raise ValueError(
                                f"Incompatible selection shape {sel_shape} for output shape {shp[i]}"
                            )
                        chp.append(shp[i])
                    else:
                        chp.append(s)
                chp_t = tuple(chp)
                eshp: list[int] = list(shp[len(sel_shape) :])
                if chp_t != selected_out.shape:
                    nout = np.zeros(chp_t, dtype=selected_out.dtype)
                    nout[:] = selected_out
                    return nout, eshp
                return selected_out, eshp

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
            shp = (len(out), *eshp)

        # apply upcast:
        if upcast and out.shape != shp:
            tmp = np.zeros(shp, dtype=out.dtype)
            tmp[:] = out
            out = tmp
            del tmp

        return out
