"""Selected state/turbine tables assembled from private chunk payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.utils import write_nc

from ._dataset_output import DatasetOutput

if TYPE_CHECKING:
    from foxes.core import Algorithm


class StateTurbineTable(DatasetOutput):
    """Collect selected farm fields without retaining a farm dataset."""

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
        """Configure table selection, naming and optional NetCDF writing.

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
            Put turbine before state in fields containing both dimensions.
        to_file
            Optional NetCDF filename, written once after assembly.
        write_pars
            NetCDF verbosity defaults to the executing algorithm's level.
            Options for ``foxes.utils.write_nc``.
        kwargs
            Private result naming and common output options.
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

    def _farm_variables(self) -> tuple[str, ...]:
        """Select only the fields needed by this table's worker payload."""
        return self.fields

    def _table(self, data: Dataset) -> Dataset:
        """Validate fields and apply external names, dimension order and selection."""
        missing = set(self.fields).difference(data.data_vars)
        if missing:
            raise KeyError(f"{self.name}: Missing farm fields {sorted(missing)}")
        table = data[list(self.fields)]
        table = table.rename(
            {
                key: value
                for key, value in self.name_map.items()
                if key in table.variables or key in table.dims
            }
        )
        if self.transpose:
            table = table.transpose(
                self.name_map.get(FC.TURBINE, FC.TURBINE),
                self.name_map.get(FC.STATE, FC.STATE),
                ...,
            )
        return table.isel(self.isel).sel(self.sel)

    def process(self, algo: Algorithm, data: Dataset) -> Dataset:
        """Return the configured table and write it once when requested.

        Parameters
        ----------
        algo
            Executing algorithm; no additional science is performed.
        data
            Selected farm fields assembled in original farm/state order.

        Returns
        -------
        table
            Renamed and selected dataset, preserving scalar and single-axis fields.

        Raises
        ------
        KeyError
            If a requested field, dimension or label is unavailable.
        """
        table = self._table(data)
        if self.to_file is not None:
            options = {"verbosity": algo.verbosity, **self.write_pars}
            write_nc(table, self.get_fpath(self.to_file), **options)
        return table
