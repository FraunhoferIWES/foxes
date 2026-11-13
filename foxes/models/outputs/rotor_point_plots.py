"""Parent-only visualization of initialized rotor quadrature metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from xarray import Dataset

from foxes.utils import Dict

from ._dataset_output import DatasetOutput

if TYPE_CHECKING:
    from foxes.core import Algorithm


class RotorPointPlot(DatasetOutput):
    """Render the executing algorithm's initialized rotor design and weights."""

    def _creates_figures(self) -> bool:
        """Keep quadrature metadata available in numerical mode."""
        return self.plot or super()._creates_figures()

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

    def get_initial_output_results(
        self,
        algo: Algorithm,
        model_data: Dataset | None = None,
        farm_data: Dataset | None = None,
        point_data: Dataset | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Store parent-only rotor rendering and writing options.

        Parameters
        ----------
        algo
            The calculation algorithm.
        model_data
            Initial model dataset passed to the base output hook.
        farm_data
            Initial farm dataset passed to the base output hook.
        point_data
            Initial target dataset, or ``None`` for farm calculations.
        extra_data
            Additional non-array inputs passed to the base output hook.

        Returns
        -------
        output_results
            Run-local results containing this plot's private options, each
            referencing its original configuration object.
        """
        output_results = super().get_initial_output_results(
            algo, model_data, farm_data, point_data, extra_data
        )

        # Bypass recursive Dict conversion to retain exact option object identity.
        dict.update(
            output_results,
            {
                self.var("plot_pars"): self.plot_pars,
                self.var("write_pars"): self.write_pars,
            },
        )
        return output_results

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Move rendering and writing options off the model while it is running.

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
                plot_pars=self.plot_pars, write_pars=self.write_pars
            )
            del self.plot_pars, self.write_pars

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Restore rendering and writing options after execution.

        Parameters
        ----------
        algo
            The calculation algorithm.
        data_stash
            Model-name keyed lifecycle storage containing this plot's options.
        sel
            Selected states for this calculation, if any.
        verbosity
            The lifecycle reporting level; ``0`` is silent.
        """
        super().unset_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            stash = data_stash[self.name]
            self.plot_pars = stash.pop("plot_pars")
            self.write_pars = stash.pop("write_pars")

    def _farm_variables(self) -> tuple[str, ...]:
        """Request no scientific farm payload for rotor metadata."""
        return ()

    def process(
        self,
        algo: Algorithm,
        data: Dataset,
        output_results: Dict[str, Any],
    ) -> Any:
        """Return rotor metadata or render it without additional initialization.

        Parameters
        ----------
        algo
            Initialized algorithm whose ``rotor_model`` supplies design points.
        data
            Empty dataset; scientific state/turbine arrays are not collected.
        output_results
            Shared run results containing this plot's rendering and writing options.

        Returns
        -------
        result
            Scatter artist, or a dataset of normalized rotor-frame points and
            unit-sum quadrature weights when rendering is disabled. Both
            numerical fields have units ``1`` and own independent array copies.
        """
        plot_pars = output_results.pop(self.var("plot_pars"))
        write_pars = output_results.pop(self.var("write_pars"))
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

        options = dict(plot_pars)
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
            ax.figure.savefig(self.get_fpath(self.to_file), **write_pars)
        self._show_figures()

        return image
