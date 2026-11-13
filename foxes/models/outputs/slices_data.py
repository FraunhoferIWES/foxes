"""Several constructor-defined parallel planes in one output dataset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from .slice_data import SliceData

if TYPE_CHECKING:
    from foxes.core import Algorithm
    from foxes.core.model import LoadedData
    from foxes.utils import Dict


class SlicesData(SliceData):
    """Evaluate configured parallel planes using one engine point calculation.

    Configure all plane positions before initialization.
    """

    def __init__(
        self, positions: Sequence[float], plane: str = "xy", **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        positions
            Fixed coordinates in metres along the axis perpendicular to ``plane``.
        plane
            ``"xy"``, ``"xz"`` or ``"yz"``.
        kwargs
            Slice fields, reduction, grid and output options. The result is xarray.
        """
        super().__init__(plane=plane, data_format="xarray", **kwargs)
        self.positions = tuple(positions)

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 0,
    ) -> None:
        """Load parallel grids into shared algorithm-owned extra data.

        Parameters
        ----------
        algo
            Initialized algorithm supplying farm geometry and state selection.
        loaded_data
            Initialization mapping whose shared grid is replaced with all
            configured planes. The resulting coordinates are read-only.
        force
            Reload existing model data.
        verbosity
            Diagnostic verbosity; zero is silent.

        Raises
        ------
        ValueError
            If no normal-axis positions were configured.
        """
        # Generate one reference plane, then translate it along its normal.
        super().load_data(algo, loaded_data, force, verbosity)
        if not self.positions:
            raise ValueError(f"{self.name}: positions must not be empty")
        points = loaded_data["extra_data"][self._grid_key]
        normal = np.array([0.0, 0.0, 1.0])
        if self.plane != "xy":
            from foxes.utils import wd2uv

            x_axis = np.append(wd2uv((self.grid_pars or {}).get("x_direction", 270)), 0)
            normal = np.cross([0, 0, 1], x_axis) if self.plane == "xz" else x_axis
        current = points[0] @ normal
        translated = np.concatenate(
            [points + (position - current) * normal for position in self.positions]
        )
        translated.setflags(write=False)
        loaded_data["extra_data"][self._grid_key] = translated

    def format_payload(
        self, algo: Algorithm, payload: dict[str, Any], output_results: Dict[str, Any]
    ) -> xr.Dataset:
        """Combine configured planes and write the resulting dataset once.

        Parameters
        ----------
        algo
            Executing algorithm, available to specialized formatters.
        payload
            Combined field arrays and point coordinates, ordered by configured
            plane position. No nested scientific calculations are launched.
        output_results
            Run-local results containing private formatting and writing options.

        Returns
        -------
        data
            Dataset with the normal-coordinate dimension in ``positions`` order,
            plane dimensions and, for state-resolved results, the state axis.
        """
        # Split already computed point values, never launch nested algorithms.
        count = len(self.positions)
        size = payload["points"].shape[1] // count
        datasets = []
        for i in range(count):
            selection = slice(i * size, (i + 1) * size)
            selected = {
                **payload,
                "points": payload["points"][:, selection],
                "values": {v: d[:, selection] for v, d in payload["values"].items()},
            }
            datasets.append(self._format_data(selected, output_results))

        normal = next(axis for axis in "xyz" if axis not in self.plane)
        normal_name = output_results[self.var("label_map")].get(normal, normal)
        normal_scale = output_results[self.var("normalize")].get(normal, 1)
        result = xr.concat(
            datasets,
            xr.IndexVariable(
                normal_name,
                np.asarray(self.positions) / normal_scale,
            ),
        )
        result.attrs.pop(normal_name, None)
        result[normal_name].attrs["units"] = "1" if normal_scale != 1 else "m"

        if self.to_file is not None:
            self._write_data(
                result, algo.verbosity, output_results[self.var("write_pars")]
            )

        return result
