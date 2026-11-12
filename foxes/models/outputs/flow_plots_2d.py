"""Fully configured two-dimensional flow plotting outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV

from . import _flow_figures
from ._variable_info import get_data_info
from .slice_data import SliceData

if TYPE_CHECKING:
    from foxes.core import Algorithm


class FlowPlots2D(SliceData):
    """Collect one configured field and render mean or state-resolved flow.

    Numerical data, point grids and reductions follow ``SliceData`` without
    inheriting animation configuration or file-writing behavior.
    """

    def _creates_figures(self) -> bool:
        """Keep numerical slices enabled independently of figure suppression."""
        return self.plot or super()._creates_figures()

    def __init__(
        self,
        var: str = FV.WS,
        plane: str = "xy",
        reduction: str = "mean",
        plot: bool = True,
        ret_data: bool = False,
        plot_pars: dict[str, Any] | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure the field, plane, reduction and rendering before execution.

        Parameters
        ----------
        var
            Field to plot.
        plane
            ``"xy"``, ``"xz"`` or ``"yz"``.
        reduction
            ``"mean"`` or ``"states"``.
        plot
            Return Matplotlib figures when enabled, or prepared numerical data.
        ret_data
            Return ``(figures, data)`` instead of only figures when plotting.
        plot_pars
            Axes, labels, colours, levels, quiver, rotor overlays and rendering
            options, including ``ret_im``, ``animated`` and ``ret_state``.
            ``quiver_n`` requests the additional wind-speed/direction fields;
            farm geometry is collected only for a configured ``rotor_color``.
        show
            Display completed figures until their windows close, after writing
            configured files. Numerical mode and ``nofig`` suppress display.
        kwargs
            Slice grids, selections, normalization, paths and format options.
        """
        variables = [var]
        if plot_pars is not None and plot_pars.get("quiver_n") is not None:
            variables = list(dict.fromkeys([var, FV.WD, FV.WS]))
        super().__init__(
            variables=variables,
            plane=plane,
            reduction=reduction,
            show=show,
            **kwargs,
        )
        self.variable = var
        self.plot = plot
        self.ret_data = ret_data
        self.plot_options = {} if plot_pars is None else dict(plot_pars)

    def _farm_variables(self) -> tuple[str, ...]:
        """Retain geometry only when figures need turbine overlays."""
        if not self.plot or self.nofig or self.plot_options.get("rotor_color") is None:
            return ()
        return (FV.X, FV.Y, FV.H, FV.D, FV.YAW, FV.AMB_WD, FV.YAWM)

    def format_payload(self, algo: Algorithm, payload: dict[str, Any]) -> Any:
        """Render the configured field from run-local numerical frames.

        Parameters
        ----------
        algo
            Executing algorithm.
        payload
            Combined selected states, farm metadata, points and field arrays.

        Returns
        -------
        result
            A mean figure, list of state figures, or the selected data format.
        """
        # Rendering functions consume explicit data, not dataset-bound output objects.
        grid = self._grid_data(payload)
        grid = tuple(
            coordinate / self.normalize.get(axis, 1)
            for axis, coordinate in zip("xyz", grid[:3])
        ) + (grid[3],)
        a, b = [grid["xyz".index(axis)] for axis in self.plane]
        values = self._field_data(payload)
        states = payload["states"]
        if self.reduction == "mean":
            states = np.array(["mean"])
        data = np.stack([values[v] for v in self.fields], axis=-1).reshape(
            len(states), len(a), len(b), len(self.fields)
        )
        parameters = dict(
            var=self.variable,
            variables=list(self.fields),
            data_format="numpy",
            vmin=self.vmin.get(self.variable),
            vmax=self.vmax.get(self.variable),
            x_direction=self._grid_options.get("x_direction", 270),
        )
        if not self.plot or self.nofig:
            data = self._format_data(payload)
            parameters["data_format"] = self.data_format
            if self.to_file is not None:
                self._write_data(data, algo.verbosity)
            if self.reduction == "mean":
                return parameters, data, grid
            return parameters, data, states, grid
        farm = Dataset(
            {v: ((FC.STATE, FC.TURBINE), d) for v, d in payload["farm"].items()},
            coords={FC.STATE: states},
        )
        options = dict(self.plot_options)
        options.setdefault("vlabel", get_data_info(self.variable)[1])
        generator = getattr(_flow_figures, f"_gen_states_fig_{self.plane}")(
            farm, self, (parameters, data, states, grid), **options
        )
        figures = list(generator)
        result = figures[0] if self.reduction == "mean" else figures
        if self.to_file is not None:
            if self.reduction != "mean":
                raise ValueError(
                    f"{self.name}: Use a mean plot or FlowAnimation to write figures"
                )
            figure = figures[0][0] if isinstance(figures[0], tuple) else figures[0]
            figure.savefig(self.get_fpath(self.to_file), **self.write_pars)
        result = (result, self._format_data(payload)) if self.ret_data else result
        self._show_figures()
        return result
