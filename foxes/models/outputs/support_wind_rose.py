"""Native support-grid wind roses executed through the output lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import foxes.variables as FV
from foxes.core import Output
from foxes.utils import Dict, plot_wind_rose_bars

if TYPE_CHECKING:
    from xarray import Dataset

    from foxes.core import Algorithm, LoadedData


class SupportWindRosePlotOutput(Output):
    """Render native binned-state frequencies at every spatial support point.

    The histogram is taken directly from initialized binned states, without
    interpolating points or evaluating farm physics. Its original speed and
    direction edges, sparse-bin locations and percentage weights are preserved.
    """

    def __init__(
        self,
        plot: bool = True,
        plot_pars: dict[str, Any] | None = None,
        to_file: str | None = None,
        result_name: str = "support_rose_plot",
        **kwargs: Any,
    ) -> None:
        """Configure native support roses without retaining a states provider.

        Parameters
        ----------
        plot
            Return ``(figure, data)`` instead of only the frequency dataset.
        plot_pars
            Rendering options: ``ncols``, ``figsize``, ``title``, caller-owned
            ``fig`` and polar ``axs``. These remain parent-owned during execution.
        to_file
            Optional figure filename, resolved using the output directory.
        result_name
            Public result key used by dictionary-returning calculations.
        kwargs
            Common output paths, figure suppression and display options.
        """
        super().__init__(fdata_variables=[], tdata_variables=[], **kwargs)
        self.plot = plot
        self.plot_pars = {} if plot_pars is None else dict(plot_pars)
        self.to_file = to_file
        self.result_name = result_name

    def _creates_figures(self) -> bool:
        return self.plot or super()._creates_figures()

    def output_coords(self) -> tuple[str, ...]:
        """Return no chunk-array coordinates for this parent-owned histogram.

        Returns
        -------
        dimensions
            Empty tuple; native support coordinates belong to the result dataset.
        """
        return ()

    def get_initial_output_results(
        self,
        algo: Algorithm,
        model_data: Dataset | None = None,
        farm_data: Dataset | None = None,
        point_data: Dataset | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Build the executing algorithm's native histogram in initial results.

        Parameters
        ----------
        algo
            Algorithm whose states are ``BinnedFieldData`` or
            ``BinnedPointCloudData``.
        model_data
            Initial model arrays and support coordinates. ``None`` reads the
            initialized algorithm's public model-data interface.
        farm_data
            Initial farm dataset, unused by native support roses.
        point_data
            Initial target dataset, unused by native support roses.
        extra_data
            Native model metadata, including retained sparse-bin indices.

        Returns
        -------
        output_results
            This run's native frequency dataset and original plotting options.
            No histogram is retained on the output model.

        Raises
        ------
        TypeError
            If the algorithm does not use a supported binned states provider.
        KeyError
            If its native bins are not exactly ``WS`` and ``WD``.
        """
        from foxes.input.states.binned import BinnedFieldData, BinnedPointCloudData

        if not isinstance(algo.states, (BinnedFieldData, BinnedPointCloudData)):
            raise TypeError(
                f"{self.name}: Require BinnedFieldData or BinnedPointCloudData states"
            )
        if model_data is None:
            model_data, extra_data = algo.get_model_data()
        loaded_data: LoadedData = {
            "coords": {key: data.to_numpy() for key, data in model_data.coords.items()},
            "data_vars": {
                key: (tuple(str(dim) for dim in data.dims), data.to_numpy())
                for key, data in model_data.data_vars.items()
            },
            "extra_data": {} if extra_data is None else extra_data,
        }
        results = super().get_initial_output_results(
            algo, model_data, farm_data, point_data, extra_data
        )
        dict.update(
            results,
            {
                self.var("data"): algo.states.get_support_wind_rose_data(loaded_data),
                self.var("plot_pars"): self.plot_pars,
            },
        )
        return results

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Stash support-rose plotting options before worker serialization.

        Parameters
        ----------
        algo
            Calculation algorithm.
        data_stash
            Lifecycle stash preserving original options, or ``None``.
        sel
            Engine state selection.
        verbosity
            Lifecycle verbosity.
        """
        super().set_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            data_stash[self.name]["plot_pars"] = self.plot_pars
        del self.plot_pars

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Restore the support-rose plotting options after execution.

        Parameters
        ----------
        algo
            Calculation algorithm.
        data_stash
            Owning stash, or ``None`` for discarded options.
        sel
            Engine state selection.
        verbosity
            Lifecycle verbosity.
        """
        super().unset_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            self.plot_pars = data_stash[self.name].pop("plot_pars")

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Return native frequencies and optionally render their support roses.

        Parameters
        ----------
        algo
            Executing algorithm; no scientific calculations are launched here.
        n_chunks_states
            Scientific state-chunk count, unused for the native histogram.
        n_chunks_targets
            Scientific target-chunk count, unused for the native histogram.
        output_results
            Native frequencies and plotting options, consumed and replaced with the result.
            Frequency dimensions are ``(point, WD, WS)`` in percent; ``support``
            contains metre coordinates ``(point, xyh)``.

        Raises
        ------
        ValueError
            If supplied axes are insufficient, non-polar or on different figures.
        """
        data = output_results.pop(self.var("data"))
        plot_pars = output_results.pop(self.var("plot_pars"))
        if not self.plot or self.nofig:
            output_results[self.result_name] = data
            return
        figure = _get_figure(data, **plot_pars)
        if self.to_file is not None:
            figure.savefig(self.get_fpath(self.to_file), bbox_inches="tight")
        output_results[self.result_name] = (figure, data)
        self._show_figures()


def _get_figure(
    data: Dataset,
    *,
    ncols: int = 4,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    fig: Any = None,
    axs: Any = None,
) -> Any:
    """Render native support histograms without retaining artists on a model."""
    import matplotlib.pyplot as plt

    support = np.asarray(data["support"])
    n_points = len(support)
    if axs is None:
        ncols = max(1, min(ncols, n_points))
        nrows = int(np.ceil(n_points / ncols))
        if fig is None:
            fig = plt.figure(
                figsize=(4.0 * ncols, 4.0 * nrows) if figsize is None else figsize
            )
        axs = fig.subplots(
            nrows, ncols, squeeze=False, subplot_kw={"projection": "polar"}
        )
    axes = np.asarray(axs, dtype=object).reshape(-1)
    if len(axes) < n_points:
        raise ValueError("Support roses require at least one polar axis per point")
    if fig is None:
        fig = axes[0].figure
    if any(ax.figure is not fig or ax.name != "polar" for ax in axes):
        raise ValueError(
            "Support rose axes must be polar and share the supplied figure"
        )
    for point_i, point in enumerate(support):
        ax = axes[point_i]
        plot_wind_rose_bars(
            ax,
            data["frequency"].to_numpy()[point_i],
            np.asarray(data.attrs[f"{FV.WD}_bounds"]),
            cmap="viridis",
        )
        ax.set_title(f"x={point[0]:.0f}, y={point[1]:.0f}, h={point[2]:.0f}")
    for ax in axes[n_points:]:
        ax.set_visible(False)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
