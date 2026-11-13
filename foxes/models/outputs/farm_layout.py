"""Constructor-defined wind-farm layout outputs and exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import foxes.constants as FC
import foxes.variables as FV

from . import _layout
from ._dataset_output import DatasetOutput
from ._units import _power_unit, _set_units

if TYPE_CHECKING:
    from xarray import Dataset

    from foxes.core import Algorithm, FData, MData, TData
    from foxes.utils import Dict


class FarmLayoutOutput(DatasetOutput):
    """Render or export a layout supplied by the executing algorithm."""

    def _creates_figures(self) -> bool:
        """Suppress figure operations without disabling layout data exports."""
        return (
            self.operation in ("get_figure", "write_plot") or super()._creates_figures()
        )

    def __init__(
        self,
        operation: str = "get_figure",
        from_results: bool = False,
        results_state: int | None = None,
        D: float | None = None,
        parameters: dict[str, Any] | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure layout rendering or export without retaining a farm.

        Parameters
        ----------
        operation
            One of ``get_figure``, ``get_layout_data``, ``get_layout_dict``,
            ``get_dataframe``, ``write_plot``, ``write_xyh``, ``write_csv``
            or ``write_json``.
        from_results
            Use calculated turbine positions instead of the algorithm's layout.
        results_state
            State position to use when ``from_results`` is enabled.
        D
            Rotor diameter in metres for normalization; ``None`` uses farm types.
        parameters
            Options for the chosen operation, including axes, annotations,
            colours, boundary overlays and filenames.
        show
            Display completed layout figures until their windows close, after
            writing any configured plot file. Data-only operations do not display.
        kwargs
            Common output paths, result name and farm-variable selection.
        """
        # Retain decisions only; farms and datasets belong to algorithm/run inputs.
        super().__init__(show=show, **kwargs)
        self.operation = operation
        self.from_results = from_results
        self.results_state = results_state
        self.D = D
        self.parameters = {} if parameters is None else dict(parameters)

    def _farm_variables(self) -> tuple[str, ...]:
        """Select geometry only when read from results, plus the colour field."""
        fields = [FV.X, FV.Y, FV.H] if self.from_results else []
        color = self._color_by if self.running else self.parameters.get("color_by")
        if color is not None and color not in (FC.FARM, FC.CLUSTER):
            choice, _, variable = color.partition("_")
            fields.append(color)
            fields.append(
                variable if choice in ("mean", "sum", "min", "max") else color
            )
            if choice == "mean":
                fields.append(FV.WEIGHT)
        return tuple(dict.fromkeys(fields))

    def get_initial_output_results(
        self,
        algo: Algorithm,
        model_data: Dataset | None = None,
        farm_data: Dataset | None = None,
        point_data: Dataset | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Retain layout options and supplied axes in initial results.

        Parameters
        ----------
        algo
            Calculation algorithm.
        model_data
            Initial model dataset forwarded unchanged.
        farm_data
            Initial farm dataset forwarded unchanged.
        point_data
            Initial target dataset forwarded unchanged.
        extra_data
            Additional model inputs.

        Returns
        -------
        output_results
            Run-owned results borrowing this layout's original parameters.
        """
        results = super().get_initial_output_results(
            algo, model_data, farm_data, point_data, extra_data
        )
        dict.update(results, {self.var("parameters"): self.parameters})
        return results

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Retain only the scientific colour selector in worker models.

        Parameters
        ----------
        algo
            Calculation algorithm.
        data_stash
            Parent lifecycle stash for restoring the complete layout options.
        sel
            Engine state selection.
        verbosity
            Lifecycle verbosity.
        """
        if not self.running:
            self._color_by = self.parameters.get("color_by")
        super().set_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            data_stash[self.name]["parameters"] = self.parameters
        del self.parameters

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Restore the layout's original parameters after execution.

        Parameters
        ----------
        algo
            Calculation algorithm.
        data_stash
            Owning lifecycle stash, or ``None`` for discarded parameters.
        sel
            Engine state selection.
        verbosity
            Lifecycle verbosity.
        """
        super().unset_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            self.parameters = data_stash[self.name].pop("parameters")

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Collect selected layout geometry and reduced colour values only.

        Parameters
        ----------
        algo
            Running algorithm owning the farm.
        mdata
            Chunk metadata including its global state offset.
        fdata
            Farm-ordered geometry and optional colour inputs.
        tdata
            Target data, unused by layouts.
        model_results
            Scientific calculation returns forwarded to collection.
        chunk_results
            Shared result mapping to extend, or ``None`` to create it.

        Returns
        -------
        chunk_results
            Mapping with turbine-only geometry for the requested state and
            turbine-only colour statistics. Negative state indices retain bounded
            trailing candidates. Algorithm-owned layouts without scientific
            colours add no arrays.

        Raises
        ------
        ValueError
            If state-chunk offsets are unavailable.
        KeyError
            If a colour field or reduction is unavailable.
        """
        results = super().calculate(
            algo, mdata, fdata, tdata, model_results, chunk_results
        )
        key = self.var("data")
        if key not in results:
            return results
        data = results[key]
        start = mdata.states_i0(counter=True)
        if start is None:
            raise ValueError(
                f"{self.name}: Layout output requires a state-chunk offset"
            )
        size = len(mdata[FC.STATE])
        compact: dict[str, Any] = {"end": start + size, "geometry": {}}
        compact["geometry"] = self._geometry_chunk(data, start, size)
        compact.update(self._color_chunk(data))
        results[key] = compact
        return results

    def _color_chunk(self, data: dict[str, Any]) -> dict[str, Any]:
        """Reduce a selected color field without returning its state history.

        Parameters
        ----------
        data
            Farm fields and weights; precomputed turbine-only color fields take
            precedence over parsing a reduction prefix.

        Returns
        -------
        color
            Turbine-only ``color`` and its merge operation, or an empty mapping
            when coloring needs no scientific field. Source arrays are not mutated.

        Raises
        ------
        KeyError
            If the color selection does not resolve a supported field/reduction.
        """
        compact: dict[str, Any] = {}
        color = self._color_by
        if color is not None and color not in (FC.FARM, FC.CLUSTER):
            choice, _, variable = color.partition("_")
            operation = {
                "mean": "weights",
                "sum": "sum",
                "min": "min",
                "max": "max",
            }.get(choice)
            name = variable if operation is not None else color
            if color in data["data_vars"] and data["data_vars"][color][0] == (
                FC.TURBINE,
            ):
                name, operation = color, None
            dims, values = data["data_vars"][name]
            weights = None
            if operation == "weights":
                wdims, weights = data["data_vars"][FV.WEIGHT]
                if wdims == (FC.STATE,):
                    weights = weights[:, None]
            if operation is None and dims != (FC.TURBINE,):
                raise KeyError(f"{self.name}: Unknown color_by '{color}'")
            if operation == "weights":
                compact["color"] = np.sum(values * weights, axis=0)
            elif operation is None:
                compact["color"] = values.copy()
            else:
                compact["color"] = getattr(np, operation)(values, axis=0)
            compact["color_operation"] = operation
        return compact

    def _geometry_chunk(
        self, data: dict[str, Any], start: int, size: int
    ) -> dict[int, dict[str, np.ndarray]]:
        """Copy only geometry candidates needed for the configured result state.

        Parameters
        ----------
        data
            Selected farm payload with geometry in its declared dimensions.
        start
            Global state offset of the current chunk.
        size
            Number of states in the chunk.

        Returns
        -------
        geometry
            Global state positions mapped to turbine-only ``X``, ``Y`` and ``H``.
            Negative selection retains only bounded trailing candidates.
        """
        geometry: dict[int, dict[str, np.ndarray]] = {}
        if self.from_results and self.results_state is not None:
            position = self.results_state
            indices = (
                range(max(0, size + position), size)
                if position < 0
                else [position - start]
                if start <= position < start + size
                else []
            )
            for index in indices:
                geometry[start + index] = {
                    v: (
                        data["data_vars"][v][1][index]
                        if FC.STATE in data["data_vars"][v][0]
                        else data["data_vars"][v][1]
                    ).copy()
                    for v in (FV.X, FV.Y, FV.H)
                }
        return geometry

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Render or export compact geometry and globally merged colour values.

        Parameters
        ----------
        algo
            Algorithm supplying farm and turbine metadata.
        n_chunks_states
            Total engine state chunks.
        n_chunks_targets
            Total engine target chunks.
        output_results
            Run-local buffers, consumed and replaced with the configured layout
            data, axes or writer result.

        Raises
        ------
        ValueError
            If required farm results or the result-state position are absent.
        IndexError
            If the requested state position is outside the selected results.
        """
        from xarray import Dataset

        parameters = output_results.pop(self.var("parameters"))
        chunks = output_results.pop(self.var("data"), {})
        ordered = [chunks[key] for key in sorted(chunks)]
        fields: dict[str, Any] = {}
        if self.from_results:
            if self.results_state is None or not ordered:
                raise ValueError(f"{self.name}: Supply farm results and results_state")
            position = self.results_state
            if position < 0:
                position += max(c["end"] for c in ordered)
            geometry = {i: row for c in ordered for i, row in c["geometry"].items()}
            if position not in geometry:
                raise IndexError(
                    f"{self.name}: Layout state {self.results_state} is unavailable"
                )
            fields.update(
                {v: ((FC.TURBINE,), d) for v, d in geometry[position].items()}
            )
        color = parameters.get("color_by")
        if ordered and "color" in ordered[0]:
            if color is None:
                raise ValueError(f"{self.name}: Colour statistics require color_by")
            operation = ordered[0]["color_operation"]
            parts = [c["color"] for c in ordered]
            if operation is None:
                values = parts[0]
            elif operation in ("sum", "weights"):
                values = np.sum(parts, axis=0)
            else:
                values = getattr(np, operation)(parts, axis=0)
            fields[color] = ((FC.TURBINE,), values)
        data = Dataset(fields)
        power_unit = (
            _power_unit(algo)
            if any(FV.amb2var.get(v, v) in (FV.P, FV.CAP, FV.MAX_P) for v in fields)
            else None
        )
        _set_units(data, power_unit)
        output_results[self.result_name] = self.process(algo, data, parameters)

    def process(
        self, algo: Algorithm, data: Dataset, parameters: dict[str, Any]
    ) -> Any:
        """Render or export the configured layout from run-local inputs.

        Parameters
        ----------
        algo
            Algorithm owning the farm and turbine metadata.
        data
            Turbine-only selected geometry and globally reduced colour fields.
        parameters
            Layout operation options, including caller-supplied axes.

        Returns
        -------
        result
            Configured layout data or Matplotlib axes; writers return ``None``.

        Raises
        ------
        ValueError
            If the operation is unknown or required result data is absent.
        """
        # Dispatch only the public layout operations rather than arbitrary methods.
        operations = {
            name: getattr(_layout, "_" + name)
            for name in (
                "get_figure",
                "get_layout_data",
                "get_layout_dict",
                "get_dataframe",
                "write_plot",
                "write_xyh",
                "write_csv",
                "write_json",
            )
        }
        if self.operation not in operations:
            raise ValueError(
                f"{self.name}: Unknown layout operation '{self.operation}'"
            )
        if self.from_results and (not data.data_vars or self.results_state is None):
            raise ValueError(f"{self.name}: Supply farm results and results_state")
        result = operations[self.operation](algo.farm, data, algo, self, **parameters)
        if self.operation == "get_figure":
            self._show_figures()
        return result
