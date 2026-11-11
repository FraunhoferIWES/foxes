"""Constructor-defined evaluation of collected farm results."""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

import foxes.constants as FC
import foxes.variables as FV

from . import _farm_evaluation as evaluate
from . import _farm_reduction as reduction
from ._dataset_output import DatasetOutput

if TYPE_CHECKING:
    from xarray import Dataset

    from foxes.core import Algorithm, FData, MData, TData
    from foxes.core.model import LoadedData
    from foxes.utils import Dict


class FarmResultsEval(DatasetOutput):
    """Evaluate a configured farm operation using compact worker statistics."""

    def __init__(
        self,
        operation: str = "reduce_states",
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure a farm evaluation without retaining calculation data.

        Parameters
        ----------
        operation
            Evaluation operation, for example ``reduce_states``,
            ``calc_farm_mean``, ``calc_yield`` or ``add_efficiency``.
            ``add_*`` operations return a new dataset with the added fields.
        parameters
            Arguments for the selected evaluation, including variable selections,
            reduction mappings, ambient selection and writing options.
        kwargs
            Common output paths, result name and farm-variable selection.
        """
        # Keep operation configuration small; arrays arrive through chunk inputs.
        super().__init__(**kwargs)
        self.operation = operation
        self.parameters = {} if parameters is None else dict(parameters)

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """Validate the operation and initialize its scientific input selection.

        Parameters
        ----------
        algo
            Algorithm supplying farm metadata and scientific fields.
        loaded_data
            Existing model data to extend, or ``None`` to create it.
        force
            Reinitialize an already initialized output.
        verbosity
            Initialization verbosity; zero is silent.

        Returns
        -------
        loaded_data
            Model data including initialized output dependencies.

        Raises
        ------
        ValueError
            If the configured operation is unknown.
        TypeError
            If the configured parameters do not match its signature.
        """
        function = getattr(evaluate, "_" + self.operation, None)
        if function is None or not callable(function):
            raise ValueError(f"{self.name}: Unknown evaluation '{self.operation}'")
        signature(function).bind(None, algo, self, **self.parameters)
        return super().initialize(algo, loaded_data, force, verbosity)

    def _farm_variables(self) -> tuple[str, ...] | None:
        """Select required fields; metadata-only evaluations send no arrays."""
        if (
            self.operation == "calc_farm_yield"
            and self.parameters.get("turbine_yield") is not None
        ):
            return ()
        return reduction.required_fields(self.operation, self.parameters)

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Calculate only statistics required by the configured farm result.

        Parameters
        ----------
        algo
            Running algorithm supplying turbine metadata and power units.
        mdata
            Chunk metadata and state coordinates.
        fdata
            Farm-ordered scientific inputs.
        tdata
            Supplied target data, unused by farm evaluations.
        model_results
            Scientific calculation returns forwarded to collection.
        chunk_results
            Shared result mapping to extend, or ``None`` to create it.

        Returns
        -------
        chunk_results
            Mapping containing private reduced values and, for means/deviations,
            valid counts and centred moments. State reductions retain turbine
            vectors; scalar sums retain scalars. Dataset operations retain their
            requested state-resolved fields. Metadata-only operations add no arrays.

        Raises
        ------
        KeyError
            If a required field or configured reduction is unavailable.
        AssertionError
            If a required field has NaNs at nonzero-weight positions.
        """
        results = super().calculate(
            algo, mdata, fdata, tdata, model_results, chunk_results
        )
        key = self.var("data")
        if (
            key in results
            and self.operation
            in reduction.STATE_OPERATIONS | reduction.TURBINE_OPERATIONS
        ):
            units = (
                evaluate._get_power_units(None, algo, self)
                if self.operation == "calc_farm_yield"
                else None
            )
            results[key] = reduction.reduce_farm_chunk(
                self.operation, self.parameters, results[key], units
            )
        return results

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Combine statistics without reconstructing eliminated dimensions.

        Parameters
        ----------
        algo
            Algorithm supplying final capacity and power-unit metadata.
        n_chunks_states
            Total engine state chunks.
        n_chunks_targets
            Total engine target chunks; farm data is collected once per state slab.
        output_results
            Run-local buffers, consumed and replaced with the configured scalar,
            mapping, dataframe, dataset or frame generator.

        Raises
        ------
        ValueError
            If required farm chunks are absent, capacity is not positive, or
            non-datetime yield data lacks an explicit duration.
        KeyError
            If yield duration or power-unit arguments conflict.
        """
        if (
            self.operation
            not in reduction.STATE_OPERATIONS | reduction.TURBINE_OPERATIONS
            or self._farm_variables() == ()
        ):
            super().finalize_results(
                algo, n_chunks_states, n_chunks_targets, output_results
            )
            return
        chunks = output_results.pop(self.var("data"), {})
        ordered = [chunks[key] for key in sorted(chunks)]
        values, index = reduction.merge_farm_chunks(self.operation, ordered)
        output_results[self.result_name] = self._reduced_result(
            algo, values, index, ordered
        )

    def _reduced_result(
        self,
        algo: Algorithm,
        values: dict[str, np.ndarray],
        index: np.ndarray,
        chunks: list[dict[str, Any]],
    ) -> Any:
        """Format merged statistics without recreating eliminated dimensions.

        Parameters
        ----------
        algo
            Algorithm supplying capacity and turbine power-unit metadata.
        values
            Globally merged field values, as turbine vectors or farm scalars.
        index
            Surviving turbine or state coordinates for dataframe results.
        chunks
            Ordered reduced payloads with scalar flags and duration metadata.

        Returns
        -------
        result
            Configured dataframe, variable-to-scalar mapping, farm metric or yield.

        Raises
        ------
        ValueError
            If farm capacity is nonpositive or the reduced operation is unknown.
        """
        operation, parameters = self.operation, self.parameters
        if operation in reduction.TURBINE_OPERATIONS or operation in (
            "reduce_states",
            "calc_states_mean",
            "calc_states_sum",
            "calc_states_std",
        ):
            result = pd.DataFrame(values, index=index)
            result.index.name = (
                FC.STATE if operation in reduction.TURBINE_OPERATIONS else FC.TURBINE
            )
            return result
        if operation in ("reduce_all", "calc_farm_mean", "calc_farm_sum"):
            return self._farm_totals(values, chunks)
        variable = FV.AMB_P if parameters.get("ambient", False) else FV.P
        power = np.sum(values[variable])
        if operation == "calc_mean_farm_power":
            return power
        if operation == "calc_farm_efficiency":
            return np.minimum(
                np.sum(values[FV.P]) / np.maximum(np.sum(values[FV.AMB_P]), 1e-14), 1
            )
        if operation == "calc_farm_capacity_factor":
            capacity = (
                np.sum(chunks[0]["capacity"])
                if "capacity" in chunks[0]
                else np.sum(algo.farm.get_capacity_array(algo))
            )
            if capacity <= 0.0:
                raise ValueError("Farm capacity must be positive")
            return power / capacity
        if operation in ("calc_yield", "calc_farm_yield"):
            return self._yield_result(algo, values[variable], index, chunks)
        raise ValueError(f"{self.name}: Unknown reduced evaluation '{operation}'")

    def _farm_totals(
        self, values: dict[str, np.ndarray], chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Finish farm totals without reducing already scalar worker statistics.

        Parameters
        ----------
        values
            Globally merged field statistics.
        chunks
            Reduced payloads identifying fields already collapsed across turbines.

        Returns
        -------
        totals
            Selected variables mapped to their final scalar farm values.
        """
        operation, parameters = self.operation, self.parameters
        choices = parameters.get(
            "turbines_op",
            {v: "mean" if operation == "calc_farm_mean" else "sum" for v in values},
        )
        scalar = chunks[0]["scalar"]
        return {
            **{v: values[v][()] for v in choices if v in scalar},
            **reduction.reduce_turbine_values(
                values, {v: c for v, c in choices.items() if v not in scalar}
            ),
        }

    def _yield_result(
        self,
        algo: Algorithm,
        mean: np.ndarray,
        index: np.ndarray,
        chunks: list[dict[str, Any]],
    ) -> Any:
        """Apply duration, power units and optional farm uncertainty once.

        Parameters
        ----------
        algo
            Algorithm supplying turbine power-unit conversions.
        mean
            State-weighted turbine power, or the unit-converted farm scalar.
        index
            Turbine coordinates for per-turbine yield results.
        chunks
            State-ordered reduced payloads carrying duration endpoints.

        Returns
        -------
        yield_result
            Per-turbine yield dataframe, farm yield scalar, or the
            ``(P50, P75, P90)`` farm-yield tuple when uncertainty is configured.
            Energy is expressed in gigawatt-hours.

        Raises
        ------
        KeyError
            If explicit power units conflict with the executing algorithm.
        """
        parameters = dict(self.parameters)
        if self.operation == "calc_farm_yield":
            parameters.setdefault("annual", True)
            for name in ("turbine_yield", "power_uncert"):
                parameters.pop(name, None)
        signature(evaluate._calc_yield).bind(None, algo, self, **parameters)
        if parameters.get("P_unit_W") is not None:
            raise KeyError("Expecting either algorithm or 'P_unit_W'")
        units = (
            np.ones_like(mean)
            if self.operation == "calc_farm_yield"
            else evaluate._get_power_units(None, algo, self)
        )
        times = np.concatenate([c["times"] for c in chunks])
        result = reduction.yield_from_mean(mean, times, units, parameters)
        name = FV.AMB_YLD if parameters.get("ambient", False) else FV.YLD
        if self.operation == "calc_yield":
            frame = pd.DataFrame({name: result}, index=index)
            frame.index.name = FC.TURBINE
            return frame
        total = np.sum(result)
        uncertainty = self.parameters.get("power_uncert")
        if uncertainty is not None:
            return (
                total,
                total * (1 - 0.675 * uncertainty),
                total * (1 - 1.282 * uncertainty),
            )
        return total

    def process(self, algo: Algorithm, data: Dataset) -> Any:
        """Evaluate the configured operation on run-local collected data.

        Parameters
        ----------
        algo
            Algorithm supplying capacity and turbine power units.
        data
            Collected farm results, owned by this output run.

        Returns
        -------
        result
            The configured scalar, mapping, dataframe, dataset or frame generator.

        Raises
        ------
        ValueError
            If the configured operation is unknown.
        """
        # Only explicitly exported numerical operations may be selected.
        operations = {
            name.removeprefix("_"): function
            for name, function in vars(evaluate).items()
            if name.startswith(
                ("_calc_", "_reduce_", "_add_", "_get_", "_gen_", "_write_")
            )
            and callable(function)
        }
        if self.operation not in operations:
            raise ValueError(f"{self.name}: Unknown evaluation '{self.operation}'")
        return operations[self.operation](data, algo, self, **self.parameters)
