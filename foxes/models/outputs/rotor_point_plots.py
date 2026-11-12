"""Parent-only visualization of initialized rotor quadrature metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from xarray import Dataset

from ._dataset_output import DatasetOutput

if TYPE_CHECKING:
    from foxes.core import Algorithm


class RotorPointPlot(DatasetOutput):
    """Render the executing algorithm's initialized rotor design and weights."""

    def __init__(
        self,
        plot: bool = True,
        plot_pars: dict[str, Any] | None = None,
        to_file: str | None = None,
        write_pars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure rotor rendering without capturing a model or algorithm.

        Parameters
        ----------
        plot
            Render a scatter plot; otherwise return quadrature metadata.
        plot_pars
            ``ax``, ``fig``, ``figsize``, ``title`` and ``Axes.scatter`` options.
        to_file
            Optional figure filename, skipped in numerical/``nofig`` mode.
        write_pars
            Options for ``Figure.savefig``.
        kwargs
            Result naming, output paths and display options.
        """
        super().__init__(variables=[], **kwargs)
        self.plot = plot
        self.plot_pars = {} if plot_pars is None else dict(plot_pars)
        self.to_file = to_file
        self.write_pars = {} if write_pars is None else dict(write_pars)

    def _farm_variables(self) -> tuple[str, ...]:
        """Request no scientific farm payload for rotor metadata."""
        return ()

    def process(self, algo: Algorithm, data: Dataset) -> Any:
        """Return rotor metadata or render it without additional initialization.

        Parameters
        ----------
        algo
            Initialized algorithm whose ``rotor_model`` supplies design points.
        data
            Empty dataset; scientific state/turbine arrays are not collected.

        Returns
        -------
        result
            Scatter artist, or a dataset of normalized rotor-frame points and
            unit-sum quadrature weights when rendering is disabled. Both
            numerical fields have units ``1`` and own independent array copies.
        """
        rotor = algo.rotor_model
        points = rotor.design_points()
        weights = rotor.rotor_point_weights()

        # Numerical results must not expose mutable rotor-model storage.
        if not self.plot or self.nofig:
            return Dataset(
                {
                    "points": (("rotor_point", "xyz"), points.copy(), {"units": "1"}),
                    "weights": ("rotor_point", weights.copy(), {"units": "1"}),
                }
            )

        options = dict(self.plot_pars)
        ax = options.pop("ax", None)
        fig = options.pop("fig", None)
        figsize = options.pop("figsize", (5, 5))
        title = options.pop("title", str(rotor))

        if ax is None:
            if fig is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                ax = fig.add_subplot(111)

        # Display dimensionless quadrature weights as percentages.
        options.setdefault("cmap", "viridis_r")
        image = ax.scatter(points[:, 1], points[:, 2], c=100 * weights, **options)
        ax.add_patch(plt.Circle((0, 0), 1, color="black", fill=False, alpha=0.8))
        ax.set(xlabel="y/R", ylabel="z/R", title=title, aspect="equal")
        colorbar = ax.figure.colorbar(
            image, ax=ax, label="Point weight [%]", shrink=0.8
        )
        ticks = np.unique(weights * 100)
        if len(ticks) <= 10:
            colorbar.set_ticks(ticks, labels=[f"{w:.2f}" for w in ticks])

        if self.to_file is not None:
            ax.figure.savefig(self.get_fpath(self.to_file), **self.write_pars)
        self._show_figures()

        return image
