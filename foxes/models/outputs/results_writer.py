"""Constructor-defined single-file CSV and NetCDF farm writers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from xarray import Dataset

import foxes.constants as FC
from foxes.utils import write_nc

from ._dataset_output import DatasetOutput

if TYPE_CHECKING:
    from foxes.core import Algorithm


class ResultsWriter(DatasetOutput):
    """Collect selected fields and write one file during parent finalization.

    Single-file table formats need their selected state history. For bounded
    NetCDF files use the count, chunk or period farm-result writers instead.
    """

    def __init__(
        self,
        file_name: str = "farm_results.csv",
        variables: list[str] | dict[str, str] | None = None,
        turbine_names: bool = False,
        state_turbine_table: bool = False,
        write_pars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure a CSV or NetCDF file without capturing datasets or frames.

        Parameters
        ----------
        file_name
            Destination with ``.csv`` or ``.nc`` suffix.
        variables
            Selected fields, or FOXES-to-external names including dimension names.
            ``None`` collects available fields without requesting derived metrics.
        turbine_names
            Use original farm turbine names instead of numeric turbine coordinates.
        state_turbine_table
            CSV only: write one selected variable in turbine columns.
        write_pars
            Options for ``Output.write`` or ``foxes.utils.write_nc``.
            NetCDF verbosity defaults to the executing algorithm's level.
        kwargs
            Result naming, paths and common output options.

        Raises
        ------
        ValueError
            If the suffix or wide-table variable selection is invalid.
        """
        suffix = Path(file_name).suffix.lower()
        if suffix not in (".csv", ".nc"):
            raise ValueError("ResultsWriter: file_name must end in .csv or .nc")
        names = dict(variables) if isinstance(variables, dict) else {}
        fields = (
            None
            if variables is None
            else [v for v in variables if v not in (FC.STATE, FC.TURBINE, FC.TNAME)]
        )
        if (
            fields == []
            or state_turbine_table
            and (suffix != ".csv" or fields is None or len(fields) != 1)
        ):
            raise ValueError("ResultsWriter: Wide CSV requires exactly one field")
        super().__init__(variables=fields, **kwargs)
        self.file_name = file_name
        self.fields = None if fields is None else tuple(fields)
        self.name_map = names
        self.turbine_names = turbine_names
        self.state_turbine_table = state_turbine_table
        self.write_pars = {} if write_pars is None else dict(write_pars)

    def _farm_variables(self) -> tuple[str, ...] | None:
        """Transport only configured fields, never dataframe/index copies."""
        return self.fields

    def process(self, algo: Algorithm, data: Dataset) -> Path:
        """Write the selected arrays once and return the resulting file path.

        Parameters
        ----------
        algo
            Algorithm supplying original farm turbine names.
        data
            State-ordered farm fields assembled from private worker payloads.

        Returns
        -------
        path
            Written CSV or NetCDF path.

        Raises
        ------
        KeyError
            If a configured variable is unavailable.
        """
        if self.fields is not None:
            data = data[list(self.fields)]
        if self.turbine_names:
            names = np.asarray(algo.farm.turbine_names)
            data = data.assign_coords(
                {FC.TURBINE: names[data[FC.TURBINE].to_numpy().astype(int)]}
            )
        renamed = data.rename(
            {key: value for key, value in self.name_map.items() if key in data}
        )
        if Path(self.file_name).suffix.lower() == ".nc":
            options = {"verbosity": algo.verbosity, **self.write_pars}
            write_nc(renamed, self.get_fpath(self.file_name), **options)
        else:
            frame, formats = self._csv_data(renamed)
            options = dict(self.write_pars)
            formats.update(options.pop("format_col2var", {}))
            self.write(self.file_name, frame, format_col2var=formats, **options)
        return self.get_fpath(self.file_name)

    def _csv_data(self, data: Dataset) -> tuple[pd.DataFrame, dict[str, str]]:
        """Format long or wide rows without mutating caller configuration/data."""
        state = self.name_map.get(FC.STATE, FC.STATE)
        turbine = self.name_map.get(FC.TURBINE, FC.TURBINE)
        formats = {value: key for key, value in self.name_map.items()}
        frame = data.to_dataframe().reorder_levels([state, turbine])
        if self.state_turbine_table:
            if self.fields is None:
                raise ValueError(f"{self.name}: Wide CSV requires one selected field")
            variable = self.name_map.get(self.fields[0], self.fields[0])
            frame = (
                frame[variable]
                .unstack(turbine)
                .reindex(index=data[state].to_numpy(), columns=data[turbine].to_numpy())
            )
            frame.columns = [
                f"{variable}_{t}" if self.turbine_names else f"{variable}_T{t:04d}"
                for t in frame.columns
            ]
            formats.update({column: self.fields[0] for column in frame.columns})
        return frame, formats
