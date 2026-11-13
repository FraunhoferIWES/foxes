from __future__ import annotations

import numpy as np
from abc import ABC
from itertools import count
from typing import TYPE_CHECKING, Any, Iterator, Literal, TypedDict, overload

import foxes.constants as FC


if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class LoadedData(TypedDict):
    """
    Data loaded during model initialization.

    Attributes
    ----------
    coords
        Coordinate values by coordinate name.
    data_vars
        Dimension names and values by data-variable name.
    extra_data
        Additional non-array data by name.
    """

    coords: dict[
        str,
        np.ndarray[Any, Any]
        | list[str]
        | list[int]
        | tuple[tuple[str, ...], np.ndarray[Any, Any]],
    ]
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray[Any, Any]]]
    extra_data: dict[str, Any]


class Model(ABC):
    """Base class for all models."""

    _ids: dict[str, Iterator[int]] = {}

    def __init__(self) -> None:
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
        verbosity
            The verbosity level; ``0`` is silent.

        """
        if self.running:
            raise ValueError(
                f"Model '{self.name}': Cannot call set_running while running"
            )
        for m in self.sub_models():
            if not m.running:
                m.set_running(algo, data_stash, sel, verbosity=verbosity)

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
            Submodels absent from this stash belong to another active lifecycle
            and remain running.
        sel
            The subset selection dictionary.
        verbosity
            The verbosity level; ``0`` is silent.

        """
        if not self.running:
            raise ValueError(
                f"Model '{self.name}': Cannot call unset_running when not running"
            )
        for m in self.sub_models():
            if m.running and (data_stash is None or m.name in data_stash):
                m.unset_running(algo, data_stash, sel, verbosity=verbosity)

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
        lookup: str = "smft",
        mdata: MData | None = None,
        fdata: FData | None = None,
        tdata: TData | None = None,
        downwind_index: int | None = None,
        accept_none: Literal[False] = False,
        accept_nan: bool = True,
        algo: Algorithm | None = None,
        selection: np.ndarray[Any, Any]
        | tuple[Any, ...]
        | list[Any]
        | slice
        | None = None,
    ) -> np.ndarray[Any, Any]: ...

    @overload
    def get_data(
        self,
        variable: str,
        target: str,
        lookup: str = "smft",
        mdata: MData | None = None,
        fdata: FData | None = None,
        tdata: TData | None = None,
        downwind_index: int | None = None,
        accept_none: Literal[True] = True,
        accept_nan: bool = True,
        algo: Algorithm | None = None,
        selection: np.ndarray[Any, Any]
        | tuple[Any, ...]
        | list[Any]
        | slice
        | None = None,
    ) -> np.ndarray[Any, Any] | None: ...

    def get_data(
        self,
        variable: str,
        target: str,
        lookup: str = "smft",
        mdata: MData | None = None,
        fdata: FData | None = None,
        tdata: TData | None = None,
        downwind_index: int | None = None,
        accept_none: bool = False,
        accept_nan: bool = True,
        algo: Algorithm | None = None,
        selection: np.ndarray[Any, Any]
        | tuple[Any, ...]
        | list[Any]
        | slice
        | None = None,
    ) -> np.ndarray[Any, Any] | None:
        """
        Retrieve data from the first matching source in lookup order.

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
            Return ``None`` if no source provides the variable.
        accept_nan
            Allow NaN values in the returned selection.
        algo
            The algorithm, needed for data from previous iterations.
        selection
            Index the full target shape with a boolean array, a slice, or a
            tuple/list of NumPy indices. Integer indices remove dimensions;
            slices retain their selected lengths. Lists are treated as tuples.

        Returns
        -------
        data
            The requested array, or ``None`` when missing and ``accept_none``
            is enabled. Named source dimensions are aligned to ``target``;
            model attributes use leading target dimensions. Shape mismatches
            are expanded to the full target shape with ``numpy.broadcast_to``
            before selection. Expanded views are read-only; unchanged source
            arrays and advanced-indexing copies retain their writeability.
            Call ``copy()`` when independent writable storage is required.

        Raises
        ------
        ValueError
            If lookup contains an unknown source, data has an incompatible
            shape, a required variable is
            missing, NaNs are disallowed, or iteration data is unavailable.
        KeyError
            If the target is unknown or required dimension sizes or iteration
            metadata cannot be determined.
        TypeError
            If the selection has an unsupported type.

        """
        # Validate options before accessing data sources.
        if any(s not in "smftw" for s in lookup):
            raise ValueError(f"Model '{self.name}': Invalid lookup '{lookup}'")

        def _geta(a: str) -> Any:
            sources: list[Any] = [
                s for s in [mdata, fdata, tdata, algo, self] if s is not None
            ]
            for s in sources:
                if a == "states_i0":
                    get_states_i0 = getattr(s, "states_i0", None)
                    if callable(get_states_i0):
                        out = get_states_i0(counter=True)
                        if out is not None:
                            return out
                else:
                    out = getattr(s, a, None)
                    if out is not None:
                        return out
            raise KeyError(
                f"Model '{self.name}': Failed to determine '{a}'. Maybe add to arguments of get_data: mdata, fdata, tdata, algo?"
            )

        dims: tuple[str, ...]
        if target == FC.STATE_TURBINE:
            if downwind_index is not None:
                raise ValueError(
                    f"Target '{target}' is incompatible with downwind_index (here {downwind_index})"
                )
            dims = (FC.STATE, FC.TURBINE)
        elif target == FC.STATE_TARGET:
            dims = (FC.STATE, FC.TARGET)
        elif target == FC.STATE_TARGET_TPOINT:
            dims = (FC.STATE, FC.TARGET, FC.TPOINT)
        else:
            raise KeyError(
                f"Model '{self.name}': Wrong parameter 'target = {target}'. Choices: {FC.STATE_TURBINE}, {FC.STATE_TARGET}, {FC.STATE_TARGET_TPOINT}"
            )

        def _match_shape(a: Any) -> np.ndarray[Any, Any]:
            out = np.asarray(a)
            if out.ndim > len(shp) or any(
                size not in (1, shp[i]) for i, size in enumerate(out.shape)
            ):
                raise ValueError(
                    f"Model '{self.name}': Shape mismatch for '{variable}': Got {out.shape}, expecting {shp}"
                )
            return out.reshape(out.shape + (1,) * (len(shp) - out.ndim))

        def _filter_dims(
            source: MData | FData | TData,
        ) -> np.ndarray[Any, Any] | None:
            a = source[variable]
            a_dims = tuple(source.dims[variable])
            if downwind_index is not None and FC.TURBINE in a_dims:
                slc = tuple(
                    [downwind_index if dd == FC.TURBINE else np.s_[:] for dd in a_dims]
                )
                a = a[slc]
                a_dims = tuple(dd for dd in a_dims if dd != FC.TURBINE)
            if len(set(a_dims)) != len(a_dims) or any(d not in dims for d in a_dims):
                return None
            order = tuple(a_dims.index(d) for d in dims if d in a_dims)
            a = np.transpose(a, order)
            sizes = dict(zip((d for d in dims if d in a_dims), a.shape))
            shape = tuple(sizes.get(d, 1) for d in dims)
            return a.reshape(shape)

        out = None
        source_key = None
        for s in lookup:
            # lookup self:
            if s == "s" and hasattr(self, variable):
                a = getattr(self, variable)
                if a is not None:
                    out = np.asarray(a)

            # lookup mdata:
            elif s == "m" and mdata is not None and variable in mdata:
                out = _filter_dims(mdata)

            # lookup fdata:
            elif s == "f" and fdata is not None and variable in fdata:
                out = _filter_dims(fdata)

            # lookup target data:
            elif (
                s == "t"
                and target != FC.STATE_TURBINE
                and tdata is not None
                and variable in tdata
            ):
                out = _filter_dims(tdata)

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
                out = np.asarray(
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
                source_key = s
                break

        # check for None:
        if out is None:
            if not accept_none:
                raise ValueError(
                    f"Model '{self.name}': Variable '{variable}' is requested but not found."
                )
            return out
        # Resolve sizes only after finding a value, including optional lookups.
        sizes = {
            FC.STATE: "n_states",
            FC.TURBINE: "n_turbines",
            FC.TARGET: "n_targets",
            FC.TPOINT: "n_tpoints",
        }
        shp = tuple(_geta(sizes[d]) for d in dims)
        out = _match_shape(out)

        # data from other chunks, only with iterations:
        if (
            target in [FC.STATE_TARGET, FC.STATE_TARGET_TPOINT]
            and source_key == "f"
            and fdata is not None
            and variable in fdata
            and tdata is not None
            and FC.STATES_SEL in tdata
        ):
            if downwind_index is None:
                raise KeyError(
                    f"Model '{self.name}': Require downwind_index for obtaining results from previous iteration"
                )
            if algo is None:
                raise ValueError(
                    f"Model '{self.name}': Iteration data found for variable '{variable}', requiring algo"
                )

            out = _match_shape(
                algo.wake_frame.get_wake_modelling_data(
                    algo, variable, downwind_index, fdata, tdata, target
                )
            )

        # Expand before indexing so NumPy determines the selected shape.
        if out.shape != shp:
            out = np.broadcast_to(out, shp)
        if selection is not None:
            if not isinstance(selection, (np.ndarray, tuple, list, slice)) or (
                isinstance(selection, np.ndarray) and selection.dtype != bool
            ):
                raise TypeError(
                    f"Model '{self.name}': Expecting a boolean array, slice, tuple or list selection, got {type(selection).__name__}"
                )
            index = tuple(selection) if isinstance(selection, list) else selection
            out = np.asarray(out[index])

        # Validate only values that are actually returned.
        if (
            not accept_nan
            and np.issubdtype(out.dtype, np.inexact)
            and np.any(np.isnan(out))
        ):
            raise ValueError(
                f"Model '{self.name}': Requested variable '{variable}' contains NaN values."
            )
        return out
