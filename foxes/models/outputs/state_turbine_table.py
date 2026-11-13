"""Selected state/turbine tables from supplied results or chunk payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.utils import Dict, write_nc

from ._dataset_output import DatasetOutput
from ._units import _normalize_farm_power, _set_units

if TYPE_CHECKING:
    from foxes.core import Algorithm
    from ._result_data import _Results


class StateTurbineTable(DatasetOutput):
    """Select supplied farm fields directly, or collect fresh scientific chunks.

    Configure table selection, naming and optional NetCDF writing.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        name_map: dict[str, str] | None = None,
        isel: dict[str, Any] | None = None,
        sel: dict[str, Any] | None = None,
        transpose: bool = False,
        to_file: str | None = None,
        write_pars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        variables
            Farm fields; ``None`` selects power and rotor-effective wind speed.
        name_map
            Mapping from FOXES fields and dimensions to external names.
        isel
            Positional selection using the external dimension names.
        sel
            Label selection after ``isel``, using external names.
        transpose
            Put retained turbine/farm/cluster axes before the state axis.
        to_file
            Optional NetCDF filename, written once after assembly.
        write_pars
            NetCDF verbosity defaults to the executing algorithm's level.
            Options for ``foxes.utils.write_nc``.
        kwargs
            Private result naming, ``aggregate_turbines``, ``aggregate_states``
            and common output options. Reductions run on workers before table
            formatting. Select input states through ``calc_outputs(states_sel=...)``
            before reducing their dimension. Without reductions, supplied farm
            results are formatted directly on the parent without engine chunks.
        """
        fields = [FV.P, FV.REWS] if variables is None else list(variables)
        if not fields or any(not isinstance(v, str) or not v for v in fields):
            raise ValueError("StateTurbineTable: variables must contain field names")
        super().__init__(variables=fields, **kwargs)
        self.fields = tuple(fields)
        self.name_map = {} if name_map is None else dict(name_map)
        self.isel = {} if isel is None else dict(isel)
        self.sel = {} if sel is None else dict(sel)
        self.transpose = transpose
        self.to_file = to_file
        self.write_pars = {} if write_pars is None else dict(write_pars)

    @property
    def requires_chunk_calculation(self) -> bool:
        """Whether this table needs chunk reductions of supplied results.

        Returns
        -------
        required
            ``True`` for turbine or state aggregation; otherwise ``False``.
        """
        return bool(self.aggregate_turbines or self.aggregate_states)

    def get_initial_output_results(
        self,
        algo: Algorithm,
        model_data: Dataset | None = None,
        farm_data: Dataset | None = None,
        point_data: Dataset | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Store parent-only table formatting options before worker execution.

        Parameters
        ----------
        algo
            The calculation algorithm.
        model_data
            Initial model dataset passed to the base output hook.
        farm_data
            Selected initial farm dataset, borrowed for parent-only processing
            when no aggregation is requested.
        point_data
            Initial target dataset, or ``None`` for farm calculations.
        extra_data
            Additional non-array inputs passed to the base output hook.

        Returns
        -------
        output_results
            Run-local results containing this table's private selection and
            writing keys, each referencing its original configuration object,
            and any supplied farm dataset needed for parent-only processing.
        """
        output_results = super().get_initial_output_results(
            algo, model_data, farm_data, point_data, extra_data
        )

        # Bypass recursive Dict conversion to retain exact configuration objects.
        dict.update(
            output_results,
            {
                self.var("name_map"): self.name_map,
                self.var("isel"): self.isel,
                self.var("sel"): self.sel,
                self.var("write_pars"): self.write_pars,
            },
        )
        if not self.requires_chunk_calculation and farm_data is not None:
            output_results[self.var("farm_data")] = farm_data
        return output_results

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Move table formatting options off the model while it is running.

        Parameters
        ----------
        algo
            The calculation algorithm.
        data_stash
            Model-name keyed lifecycle storage, or ``None`` to retain model state.
        sel
            Selected states for this calculation, if any.
        verbosity
            The lifecycle reporting level; ``0`` is silent.
        """
        super().set_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            data_stash[self.name].update(
                name_map=self.name_map,
                isel=self.isel,
                sel=self.sel,
                write_pars=self.write_pars,
            )
            del self.name_map, self.isel, self.sel, self.write_pars

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Restore table formatting options after execution.

        Parameters
        ----------
        algo
            The calculation algorithm.
        data_stash
            Model-name keyed lifecycle storage containing this table's options.
        sel
            Selected states for this calculation, if any.
        verbosity
            The lifecycle reporting level; ``0`` is silent.
        """
        super().unset_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            stash = data_stash[self.name]
            self.name_map = stash.pop("name_map")
            self.isel = stash.pop("isel")
            self.sel = stash.pop("sel")
            self.write_pars = stash.pop("write_pars")

    def _farm_variables(self) -> tuple[str, ...]:
        """Select only the fields needed by this table's worker payload."""
        return self.fields

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Format supplied fields directly unless fresh chunks were collected.

        Parameters
        ----------
        algo
            Running algorithm supplying turbine units and verbosity.
        n_chunks_states
            Number of state chunks, zero for entirely parent-only execution.
        n_chunks_targets
            Number of target chunks.
        output_results
            Run-local mapping containing borrowed data and formatting options.
            Private entries are consumed; the public table uses ``result_name``.
        """
        farm_data = output_results.pop(self.var("farm_data"), None)
        if farm_data is None or output_results.get(self.var("data")):
            super().finalize_results(
                algo, n_chunks_states, n_chunks_targets, output_results
            )
            return

        # Reuse power conversion without copying or mutating borrowed arrays.
        output_results.pop(self.var("data"), None)
        data = farm_data[[v for v in self.fields if v in farm_data]].copy(deep=False)
        payload: _Results = {
            "coords": {
                FC.TURBINE: (
                    farm_data.get_index(FC.TURBINE).to_numpy()
                    if FC.TURBINE in farm_data.dims
                    else np.arange(algo.n_turbines)
                )
            },
            "data_vars": {
                str(k): (tuple(str(d) for d in v.dims), v.values)
                for k, v in data.data_vars.items()
            },
            "attrs": dict(data.attrs),
        }
        _normalize_farm_power(payload, algo, farm_data)
        for variable, (dims, values) in payload["data_vars"].items():
            data[variable] = (dims, values, dict(data[variable].attrs))
        data.attrs.update(payload["attrs"])
        _set_units(data)
        output_results[self.result_name] = self.process(algo, data, output_results)

    def _table(self, data: Dataset, output_results: Dict[str, Any]) -> Dataset:
        """Validate fields and apply external names, dimension order and selection."""
        name_map = output_results.pop(self.var("name_map"))
        isel = output_results.pop(self.var("isel"))
        sel = output_results.pop(self.var("sel"))
        missing = set(self.fields).difference(data.data_vars)
        if missing:
            raise KeyError(f"{self.name}: Missing farm fields {sorted(missing)}")
        table = data[list(self.fields)]
        table = table.rename(
            {
                key: value
                for key, value in name_map.items()
                if key in table.variables or key in table.dims
            }
        )
        if self.transpose:
            table = table.transpose(
                *(
                    name_map.get(dim, dim)
                    for dim in (FC.TURBINE, FC.FARM, FC.CLUSTER, FC.STATE)
                    if name_map.get(dim, dim) in table.dims
                ),
                ...,
            )
        return table.isel(isel).sel(sel)

    def process(
        self, algo: Algorithm, data: Dataset, output_results: Dict[str, Any]
    ) -> Dataset:
        """Return the configured table and write it once when requested.

        Parameters
        ----------
        algo
            Executing algorithm; no additional science is performed.
        data
            Selected farm fields assembled in original farm/state order.
        output_results
            Shared run results containing this table's private formatting options.

        Returns
        -------
        table
            Renamed and selected dataset, preserving scalar and single-axis fields.

        Raises
        ------
        KeyError
            If a requested field, dimension or label is unavailable.
        """
        write_pars = output_results.pop(self.var("write_pars"))
        table = self._table(data, output_results)
        if self.to_file is not None:
            options = {"verbosity": algo.verbosity, **write_pars}
            write_nc(table, self.get_fpath(self.to_file), **options)
        return table
