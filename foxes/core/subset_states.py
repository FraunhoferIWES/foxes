from __future__ import annotations

# mypy: disable-error-code=override

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config

from .data import FData, MData, TData
from .model import Model
from .states import States

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.model import LoadedData


class SubsetStates(States):
    """
    A positional subset of another states model.

    The selected states retain their original order and weights. In
    particular, weights are not normalized to sum to one within the subset.
    """

    def __init__(
        self,
        states: States,
        states_isel: Sequence[int] | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        states
            The original states.
        states_isel
            Unique positional indices of the selected states.
        kwargs
            Additional parameters for the base class.
        """
        super().__init__(load_mode=states.load_mode, **kwargs)
        self.states = states

        indices = np.asarray(states_isel)
        if indices.ndim != 1:
            raise ValueError(
                f"States '{self.name}': Expecting one-dimensional state indices, got shape {indices.shape}"
            )
        if indices.size == 0:
            raise ValueError(f"States '{self.name}': State selection is empty")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(
                f"States '{self.name}': Expecting integer state indices, got dtype {indices.dtype}"
            )

        self.states_isel = indices.astype(config.dtype_int, copy=True)
        if len(np.unique(self.states_isel)) != len(self.states_isel):
            raise ValueError(
                f"States '{self.name}': State selection contains duplicate indices"
            )
        self._indices_validated = False
        self._validate_indices(require_size=False)

    def _validate_indices(self, require_size: bool) -> None:
        """Normalize and validate indices when the source size is available."""
        if self._indices_validated:
            return
        n_states = self.states.size()
        if n_states <= 0:
            if require_size:
                raise ValueError(
                    f"States '{self.name}': Cannot select from an empty states model"
                )
            return

        indices = self.states_isel.copy()
        indices[indices < 0] += n_states
        if np.any((indices < 0) | (indices >= n_states)):
            raise IndexError(
                f"States '{self.name}': State indices out of range for {n_states} states: {indices.tolist()}"
            )
        if len(np.unique(indices)) != len(indices):
            raise ValueError(
                f"States '{self.name}': State selection contains duplicate indices"
            )
        self.states_isel = indices
        self._indices_validated = True

    def sub_models(self) -> list[Model]:
        """Return the wrapped states model."""
        return [self.states]

    def output_point_vars(self, algo: Algorithm) -> list[str]:
        """Return the variables modified by the wrapped states model."""
        return self.states.output_point_vars(algo)

    def size(self) -> int:
        """Return the number of selected states."""
        return len(self.states_isel)

    def index(self) -> list[int]:
        """Return the selected state labels in selection order."""
        self._validate_indices(require_size=True)
        source_index = self.states.index()
        return [source_index[i] for i in self.states_isel]

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 0,
    ) -> None:
        """Create the mapping between selected and original states."""
        self._validate_indices(require_size=True)
        if force:
            super().load_data(algo, loaded_data, force=force, verbosity=verbosity)

        self.STATE0 = self.var(FC.STATE + "0")
        self.SMAP = self.var("smap")
        coords = loaded_data["coords"]
        data_vars = loaded_data["data_vars"]

        if not force and self.SMAP in data_vars:
            return

        if FC.STATE in coords:
            coords[self.STATE0] = coords.pop(FC.STATE)

        need_state0 = False
        for data_name in list(data_vars.keys()):
            dims, data = data_vars[data_name]
            if FC.STATE in dims:
                data_vars.pop(data_name)
                dims = tuple(self.STATE0 if dim == FC.STATE else dim for dim in dims)
                data_vars[data_name] = (dims, data)
                need_state0 = True

        if FV.WEIGHT not in data_vars:
            data_vars[FV.WEIGHT] = (
                (self.STATE0,),
                np.full(
                    self.states.size(),
                    1 / self.states.size(),
                    dtype=config.dtype_double,
                ),
            )
            need_state0 = True

        data_vars[self.SMAP] = ((FC.STATE,), self.states_isel.copy())
        if self.STATE0 in coords and not need_state0:
            coords.pop(self.STATE0)

    def load_chunk_data(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData | None = None,
        tdata: TData | None = None,
        *extra_data: Any,
    ) -> None:
        """Load and gather source data required by the selected-state chunk."""
        if self.load_mode == "preload":
            return
        if fdata is None or tdata is None:
            raise ValueError(
                f"States '{self.name}': Missing required fdata/tdata in load_chunk_data"
            )

        smap = mdata[self.SMAP]
        source_i0 = int(np.min(smap))
        source_i1 = int(np.max(smap)) + 1
        source_size = source_i1 - source_i0
        source_data: dict[str, Any] = {
            FC.STATE: np.arange(source_i0, source_i1, dtype=config.dtype_int)
        }
        source_dims: dict[str, tuple[str, ...]] = {FC.STATE: (FC.STATE,)}

        for data_name, data in mdata.items():
            if data_name in (FC.STATE, self.SMAP, self.STATE0):
                continue
            dims = mdata.dims[data_name]
            if dims and dims[0] == self.STATE0:
                source_data[data_name] = data[source_i0:source_i1]
                source_dims[data_name] = (FC.STATE,) + dims[1:]
            elif self.STATE0 in dims:
                raise ValueError(
                    f"States '{self.name}': Expecting {self.STATE0} at position 0 for {data_name}, got {dims}"
                )
            else:
                source_data[data_name] = data
                source_dims[data_name] = dims

        source_mdata = MData(
            data=source_data,
            dims=source_dims,
            states_i0=source_i0,
            chunki_states=mdata.chunki_states,
            chunki_points=mdata.chunki_points,
            n_chunks_states=mdata.n_chunks_states,
            n_chunks_points=mdata.n_chunks_points,
            extra_data=mdata.extra_data,
            name=f"{mdata.name}_source",
        )
        keys_before = set(source_mdata.keys())
        self.states.load_chunk_data(algo, source_mdata, fdata, tdata, *extra_data)
        source_indices = smap - source_i0

        for data_name in set(source_mdata.keys()) - keys_before:
            data = source_mdata[data_name]
            dims = source_mdata.dims[data_name]
            if dims and dims[0] == FC.STATE:
                if data.shape[0] != source_size:
                    raise ValueError(
                        f"States '{self.name}': Loaded source data '{data_name}' has {data.shape[0]} states, expected {source_size}"
                    )
                mdata[data_name] = data[source_indices]
                mdata.dims[data_name] = dims
            elif FC.STATE in dims:
                raise ValueError(
                    f"States '{self.name}': Expecting {FC.STATE} at position 0 for {data_name}, got {dims}"
                )
            else:
                mdata[data_name] = data
                mdata.dims[data_name] = dims

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData | None = None,
        fdata: FData | None = None,
        tdata: TData | None = None,
        *args: Any,
        **parameters: Any,
    ) -> dict[str, np.ndarray]:
        """Run the wrapped states model for the selected states."""
        if mdata is None or fdata is None or tdata is None:
            raise KeyError(
                f"States '{self.name}': Missing input data for calculate(), expected mdata, fdata and tdata"
            )

        super().calculate(algo, mdata, fdata, tdata)
        smap = mdata[self.SMAP]

        def _map(input_data: Any, data_class: Any) -> Any:
            mapped_data = {}
            mapped_dims = {}
            for data_name, data in input_data.items():
                if data_name not in input_data.dims:
                    continue
                dims = input_data.dims[data_name]
                if data_name in (self.SMAP, self.STATE0):
                    continue
                if dims and dims[0] == self.STATE0:
                    mapped_data[data_name] = data[smap]
                    mapped_dims[data_name] = (FC.STATE,) + dims[1:]
                elif self.STATE0 in dims:
                    raise ValueError(
                        f"States '{self.name}': Found source-state dimension away from position 0 for '{data_name}': {dims}"
                    )
                else:
                    shape = tuple(input_data.sizes[dim] for dim in dims)
                    mapped = np.broadcast_to(data, shape)
                    mapped_data[data_name] = (
                        mapped.copy() if data_class is TData else mapped
                    )
                    mapped_dims[data_name] = dims
            return data_class.from_data(
                input_data,
                data=mapped_data,
                dims=mapped_dims,
                extra_data=input_data.extra_data,
                name=input_data.name + "_subset",
            )

        source_mdata = _map(mdata, MData)
        source_fdata = _map(fdata, FData)
        source_tdata = _map(tdata, TData)
        results = self.states.calculate(
            algo, source_mdata, source_fdata, source_tdata, *args, **parameters
        )

        assert FV.WEIGHT in source_tdata, (
            f"Missing '{FV.WEIGHT}' in tdata results from states '{self.states.name}'"
        )
        results[FV.WEIGHT] = np.zeros(
            (
                source_tdata.n_states,
                source_tdata.n_targets,
                source_tdata.n_tpoints,
            ),
            dtype=config.dtype_double,
        )
        results[FV.WEIGHT][:] = source_tdata[FV.WEIGHT]
        tdata[FV.WEIGHT] = results[FV.WEIGHT]
        tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)
        for variable in results:
            if results[variable].shape[0] == 1 and source_tdata.n_states > 1:
                results[variable] = np.broadcast_to(
                    results[variable],
                    (source_tdata.n_states,) + results[variable].shape[1:],
                ).copy()
        return results
