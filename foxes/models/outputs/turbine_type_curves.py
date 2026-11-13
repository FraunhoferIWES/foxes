"""Turbine curves collected from an explicitly configured wind-speed scan."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.utils import Dict

from .state_turbine_table import StateTurbineTable
from ._variable_info import get_data_info

if TYPE_CHECKING:
    from foxes.core import Algorithm


class TurbineTypeCurves(StateTurbineTable):
    """Plot selected turbine curves without launching hidden simulations.

    Configure wind-speed scan states, turbine types and optional power-mask
    comparison turbines on the executing no-wake algorithm.

    Configure collected fields and final curve rendering.
    """

    def _creates_figures(self) -> bool:
        """Keep collected curve data available in numerical mode."""
        return self.plot or super()._creates_figures()

    def __init__(
        self,
        variables: list[str] | None = None,
        ws_var: str = FV.AMB_REWS,
        plot: bool = True,
        plot_pars: dict[str, Any] | None = None,
        to_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        variables
            Curve fields; ``None`` selects power and thrust coefficient.
        ws_var
            Wind-speed abscissa in metres per second.
        plot
            Render curves instead of returning the selected dataset.
        plot_pars
            ``axs``, ``figsize``, ``titles``, ``x_label``, ``y_labels`` and
            remaining ``Axes.plot`` options. Each selected turbine is one curve.
        to_file
            Optional figure filename, skipped in numerical/``nofig`` mode.
        kwargs
            Table selections, figure ``write_pars``, paths and display options.
            Dimension renaming and transposition are not supported.
        """
        fields = [FV.P, FV.CT] if variables is None else list(variables)
        if not fields:
            raise ValueError("TurbineTypeCurves: variables must contain field names")
        super().__init__(variables=list(dict.fromkeys([ws_var, *fields])), **kwargs)
        if self.name_map or self.transpose:
            raise ValueError(
                "TurbineTypeCurves: name_map and transpose are not supported"
            )
        self.curve_variables = tuple(fields)
        self.ws_var = ws_var
        self.plot = plot
        self.plot_pars = {} if plot_pars is None else dict(plot_pars)
        self.to_file = to_file

    def get_initial_output_results(
        self,
        algo: Algorithm,
        model_data: Dataset | None = None,
        farm_data: Dataset | None = None,
        point_data: Dataset | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Store curve plotting options alongside inherited table formatting.

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
            Run-local results containing this output's private plotting key and
            the inherited table keys, each referencing its original options.
        """
        output_results = super().get_initial_output_results(
            algo, model_data, farm_data, point_data, extra_data
        )

        # Bypass recursive Dict conversion to retain exact plotting configuration.
        dict.update(output_results, {self.var("plot_pars"): self.plot_pars})
        return output_results

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Move curve plotting options off the model while it is running.

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
            data_stash[self.name]["plot_pars"] = self.plot_pars
            del self.plot_pars

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Restore curve plotting options after execution.

        Parameters
        ----------
        algo
            The calculation algorithm.
        data_stash
            Model-name keyed lifecycle storage containing this output's options.
        sel
            Selected states for this calculation, if any.
        verbosity
            The lifecycle reporting level; ``0`` is silent.
        """
        super().unset_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            self.plot_pars = data_stash[self.name].pop("plot_pars")

    def process(
        self,
        algo: Algorithm,
        data: Dataset,
        output_results: Dict[str, Any],
    ) -> Any:
        """Render selected wind-speed curves or return their numerical dataset.

        Parameters
        ----------
        algo
            Algorithm supplying turbine names; never rerun or modified here.
        data
            Selected, chunk-collected curve fields.
        output_results
            Shared run results containing this output's private formatting options.

        Returns
        -------
        result
            One-dimensional axes array, or the dataset in numerical/``nofig`` mode.

        Raises
        ------
        ValueError
            If the number of supplied axes does not match the curve fields.
        """
        plot_pars = output_results.pop(self.var("plot_pars"))
        write_pars = output_results.pop(self.var("write_pars"))
        table = self._table(data, output_results)
        if not self.plot or self.nofig:
            return table
        options = dict(plot_pars)
        axes = options.pop("axs", None)
        size = options.pop("figsize", (5 * len(self.curve_variables), 4))
        if axes is None:
            _, axes = plt.subplots(1, len(self.curve_variables), figsize=size)
        axes = np.atleast_1d(axes).reshape(-1)
        if len(axes) != len(self.curve_variables):
            raise ValueError(f"{self.name}: One axis is required for each curve field")
        titles = options.pop("titles", {})
        labels = options.pop("y_labels", {})
        xlabel = options.pop("x_label", "Wind speed [m/s]")
        for ax, variable in zip(axes, self.curve_variables):
            for index, turbine in enumerate(table[FC.TURBINE].to_numpy()):
                curve = table.sel({FC.TURBINE: turbine})
                order = np.argsort(curve[self.ws_var].to_numpy(), kind="stable")
                ax.plot(
                    curve[self.ws_var].to_numpy()[order],
                    curve[variable].to_numpy()[order],
                    label=algo.farm.turbine_names[int(turbine)],
                    linestyle=options.get(
                        "linestyle", ("-", "--", "-.", ":")[index % 4]
                    ),
                    **{
                        key: value
                        for key, value in options.items()
                        if key != "linestyle"
                    },
                )
            ax.set(
                title=titles.get(variable, variable),
                xlabel=xlabel,
                ylabel=labels.get(
                    variable,
                    get_data_info(variable, table[variable].attrs.get("units"))[1],
                ),
            )
            ax.grid(axis="y")
            ax.legend()
        if self.to_file is not None:
            axes[0].figure.savefig(self.get_fpath(self.to_file), **write_pars)
        self._show_figures()
        return axes
