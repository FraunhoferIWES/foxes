"""Constructor-defined wind-rose frequencies and binned field plots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

import foxes.constants as FC
import foxes.variables as FV

from ._rose import _bins_get_figure, _rose_get_figure
from ._rose_output import RoseOutput
from ._variable_info import get_data_info

if TYPE_CHECKING:
    from xarray import Dataset

    from foxes.core import Algorithm, MData


class RosePlotOutput(RoseOutput):
    """Collect weighted frequencies for one constructor-defined wind rose."""

    get_data_info = staticmethod(get_data_info)

    def __init__(
        self,
        wd_sectors: int = 12,
        ws_var: str = FV.AMB_REWS,
        ws_bins: Sequence[float] = (0, 5, 10, 15, 20, 25),
        wd_var: str = FV.AMB_WD,
        turbine: int = 0,
        point: int = 0,
        use_points: bool = False,
        add_inf: bool = False,
        plot: bool = True,
        to_file: str | None = None,
        plot_pars: dict[str, Any] | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure frequency bins, data location and optional rendering.

        Parameters
        ----------
        wd_sectors
            Positive number of evenly spaced wind-direction sectors.
        ws_var
            Wind-speed field used for binning.
        ws_bins
            Strictly increasing wind-speed bin edges in metres per second.
        wd_var
            Wind-direction field in degrees.
        turbine
            Farm-order turbine index for farm data.
        point
            Point index when using supplied point results.
        use_points
            Read point rather than farm fields.
        add_inf
            Append an infinite final speed edge.
        plot
            Return polar axes and frequency data when enabled; otherwise return
            only the frequency dataset.
        to_file
            Optional figure filename.
        plot_pars
            Matplotlib axes, legend, colour and title options.
        show
            Display the completed rose until its window closes, after writing
            any configured figure file. Numerical mode and ``nofig`` suppress it.
        kwargs
            Common output paths and result-name configuration.
        """
        # Constructors retain only decisions, not states, algorithms or datasets.
        super().__init__(use_points=use_points, show=show, **kwargs)
        self.wd_sectors = wd_sectors
        self.ws_var = ws_var
        self.ws_bins = tuple(ws_bins)
        self.wd_var = wd_var
        self.turbine = turbine
        self.point = point
        self.add_inf = add_inf
        self.plot = plot
        self.to_file = to_file
        self.plot_pars = {} if plot_pars is None else dict(plot_pars)
        self.result_dimension = FC.POINT if use_points else FC.TURBINE

    def _contraction(self) -> str:
        return "frequency"

    def process(self, algo: Algorithm, data: Dataset) -> Any:
        """Create the configured frequency dataset and optional polar figure.

        Parameters
        ----------
        algo
            Executing algorithm.
        data
            Run-local frequency dataset already combined from worker histograms.

        Returns
        -------
        result
            Frequency dataset, or ``(axes, dataset)`` when plotting is enabled.

        """
        if not self.plot or self.nofig:
            return data
        axes, result = _rose_get_figure(
            data,
            self,
            ws_var=self.ws_var,
            wd_var=self.wd_var,
            ret_data=True,
            **self.plot_pars,
        )
        if self.to_file is not None:
            axes.figure.savefig(self.get_fpath(self.to_file), bbox_inches="tight")
        self._show_figures()
        return axes, result


class StatesRosePlotOutput(RosePlotOutput):
    """Evaluate the executing algorithm's states at one configured location."""

    def __init__(self, point: Sequence[float], **kwargs: Any) -> None:
        """Configure one state-evaluation location without running a calculation.

        Parameters
        ----------
        point
            Three coordinates ``(x, y, z)`` in metres.
        kwargs
            Rose binning, rendering and common output configuration.
        """
        kwargs.setdefault("ws_var", FV.AMB_WS)
        super().__init__(use_points=True, **kwargs)
        self.location = tuple(point)
        self.point = 0

    def get_calc_points(self, algo: Algorithm) -> np.ndarray:
        """Return the single configured state-evaluation point.

        Parameters
        ----------
        algo
            Initialized executing algorithm.

        Returns
        -------
        points
            Coordinates shaped ``(1, 3)``.

        Raises
        ------
        ValueError
            If the configured location does not contain three finite coordinates.
        """
        points = np.asarray(self.location, dtype=float)
        if points.shape != (3,) or not np.all(np.isfinite(points)):
            raise ValueError(f"{self.name}: Require three finite point coordinates")
        return points[None, :]

    def get_calc_points_from_chunk(self, mdata: MData) -> np.ndarray:
        """Return the configured location for bounded farm-chunk evaluation.

        Parameters
        ----------
        mdata
            Current model-data chunk; the configured location is state-independent.

        Returns
        -------
        points
            Fresh static coordinate array with shape ``(1, 3)``.
        """
        return np.asarray(self.location, dtype=float)[None, :]


class WindRoseBinPlot(RoseOutput):
    """Reduce one field within configured wind-direction and wind-speed bins."""

    def __init__(
        self,
        variable: str,
        ws_bins: Sequence[float],
        wd_sectors: int = 12,
        wd_var: str = FV.AMB_WD,
        ws_var: str = FV.AMB_REWS,
        turbine: int = 0,
        contraction: str = "weights",
        plot: bool = True,
        to_file: str | None = None,
        plot_pars: dict[str, Any] | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure the binned field, reduction and optional polar rendering.

        Parameters
        ----------
        variable
            Field to reduce in each bin.
        ws_bins
            Increasing wind-speed bin edges in metres per second.
        wd_sectors
            Number of wind-direction sectors.
        wd_var
            Wind-direction field.
        ws_var
            Wind-speed field.
        turbine
            Farm-order turbine index.
        contraction
            ``weights``, ``mean_no_weights`` or ``sum_no_weights``.
        plot
            Return axes and binned data instead of only a dataset.
        to_file
            Optional figure filename.
        plot_pars
            Axes, title, size and colour-mapping options.
        show
            Display the completed plot until its window closes, after writing
            any configured figure file. Numerical mode and ``nofig`` suppress it.
        kwargs
            Common output configuration.
        """
        super().__init__(show=show, **kwargs)
        self.variable = variable
        self.ws_bins = tuple(ws_bins)
        self.wd_sectors = wd_sectors
        self.wd_var = wd_var
        self.ws_var = ws_var
        self.turbine = turbine
        self.contraction = contraction
        self.plot = plot
        self.to_file = to_file
        self.plot_pars = {} if plot_pars is None else dict(plot_pars)

    def _contraction(self) -> str:
        return self.contraction

    def process(self, algo: Algorithm, data: Dataset) -> Any:
        """Render globally combined bin statistics without retaining raw samples.

        Parameters
        ----------
        algo
            Executing algorithm.
        data
            Globally combined direction-by-speed bin dataset.

        Returns
        -------
        result
            Bin dataset in numerical mode, or ``(axes, dataset)`` when plotting.
            A configured figure filename is written only after merging.
        """
        parameters: dict[str, Any] = dict(
            variable=self.variable,
            wd_var=self.wd_var,
            ws_var=self.ws_var,
        )
        if not self.plot or self.nofig:
            return data
        axes, result = _bins_get_figure(
            data, self, **parameters, ret_data=True, **self.plot_pars
        )
        if self.to_file is not None:
            axes.figure.savefig(self.get_fpath(self.to_file), bbox_inches="tight")
        self._show_figures()
        return axes, result
