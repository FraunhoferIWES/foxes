"""Mergeable rose histograms without state-sized worker return arrays."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

import foxes.constants as FC
import foxes.variables as FV
from foxes.utils import Dict

from ._dataset_output import DatasetOutput
from ._result_data import _Results
from ._units import _normalize_farm_power, _power_unit, _set_units

if TYPE_CHECKING:
    from foxes.core import Algorithm, FData, MData, TData
    from foxes.core.model import LoadedData


class RoseOutput(DatasetOutput):
    """Reduce selected farm/point samples into fixed-size bin accumulators."""

    ws_bins: Sequence[float]
    wd_sectors: int
    ws_var: str
    wd_var: str
    turbine: int
    point: int
    variable: str
    add_inf: bool

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """Validate bins and initialize only fields needed by the histogram.

        Parameters
        ----------
        algo
            Algorithm supplying scientific point dependencies or farm fields.
        loaded_data
            Existing model data to extend, or ``None`` to create it.
        force
            Reinitialize an already initialized output.
        verbosity
            Initialization verbosity; zero is silent.

        Returns
        -------
        loaded_data
            Initialized output and dependency data.

        Raises
        ------
        ValueError
            If direction sectors or wind-speed bin edges are invalid.
        KeyError
            If the contraction is unknown.
        """
        bins = np.asarray(self.ws_bins, dtype=float)
        if (
            not isinstance(self.wd_sectors, (int, np.integer))
            or self.wd_sectors < 1
            or bins.ndim != 1
            or len(bins) < 2
            or np.any(np.isnan(bins))
            or np.any(bins[1:] <= bins[:-1])
        ):
            raise ValueError(
                f"{self.name}: Require positive sectors and increasing bins"
            )
        if self._contraction() not in (
            "frequency",
            "weights",
            "mean_no_weights",
            "sum_no_weights",
        ):
            raise KeyError(
                f"{self.name}: Unknown rose contraction '{self._contraction()}'"
            )
        if not self.use_points:
            self._fdata_variables = list(self._farm_variables())
        return super().initialize(algo, loaded_data, force, verbosity)

    def _contraction(self) -> str:
        """Select the configured frequency or field-bin reduction.

        Returns
        -------
        contraction
            ``frequency``, ``weights``, ``mean_no_weights`` or ``sum_no_weights``.
        """
        raise NotImplementedError

    def _farm_variables(self) -> tuple[str, ...]:
        """Select the state fields required by the configured histogram.

        Returns
        -------
        variables
            Unique speed, direction and weight names, plus the binned field when used.
        """
        fields = [self.ws_var, self.wd_var, FV.WEIGHT]
        if self._contraction() != "frequency":
            fields.append(self.variable)
        return tuple(dict.fromkeys(fields))

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Accumulate bin sums and denominators without returning state samples.

        Parameters
        ----------
        algo
            Running algorithm supplying initialized scientific point models.
        mdata
            Chunk metadata and any loaded calculation points.
        fdata
            Farm-ordered fields or scientific inputs for point calculations.
        tdata
            Supplied target fields, or ``None`` for a farm run.
        model_results
            Scientific calculation returns; fields are read from data containers.
        chunk_results
            Shared result mapping to extend, or ``None`` to create it.

        Returns
        -------
        chunk_results
            Mapping with direction-by-speed bin sums and, for bin means,
            denominators. Only the selected turbine or point contributes.
            Negative point indices retain bounded trailing histogram candidates.

        Raises
        ------
        KeyError
            If a required scientific field is unavailable.
        ValueError
            If point dependencies or field dimensions are invalid.
        """
        results = Dict() if chunk_results is None else chunk_results
        if not self.use_points:
            if mdata.chunki_points not in (None, 0):
                return results

            # Normalize the selected turbine before discarding sample identities.
            samples = self._samples(fdata, self.turbine, FC.TURBINE)
            payload: _Results = {
                "coords": {FC.TURBINE: np.asarray([fdata[FC.TURBINE][self.turbine]])},
                "data_vars": {
                    variable: ((FC.STATE, FC.TURBINE), values[:, None])
                    for variable, values in samples.items()
                },
                "attrs": {},
            }
            _normalize_farm_power(
                payload, algo, mdata.extra_data.get(FC.PREV_FARM_RESULTS)
            )
            samples = {
                variable: values[:, 0]
                for variable, (_, values) in payload["data_vars"].items()
            }
            results[self.var("data")] = self._histogram(samples)
            return results
        if tdata is not None:
            if self.point < 0:
                labels = tdata[FC.TARGET]
                start = max(0, len(labels) + self.point)
                results[self.var("data")] = {
                    "last_target": int(labels[-1]),
                    "candidates": {
                        int(labels[i]): self._histogram(
                            self._samples(tdata, i, FC.TARGET)
                        )
                        for i in range(start, len(labels))
                    },
                }
                return results
            selected = self._point_sample(tdata)
            if selected is not None:
                results[self.var("data")] = self._histogram(selected)
            return results
        from ._point_calculation import point_blocks

        points = self.get_calc_points_from_chunk(mdata)
        if points is None or self._point_models is None:
            raise ValueError(
                f"{self.name}: Point rose requires initialized calculation points"
            )
        parts = [
            self._histogram(self._samples(block, 0, FC.TARGET))
            for block in point_blocks(
                algo,
                mdata,
                fdata,
                points,
                self._point_models,
                self._point_parameters,
                self._chunk_size_points,
            )
        ]
        results[self.var("data")] = self._sum_histograms(parts)
        return results

    def _samples(
        self, data: FData | TData, index: int, dimension: str
    ) -> dict[str, np.ndarray]:
        """Select scalar state fields for one turbine or target.

        Parameters
        ----------
        data
            Chunk-local farm or target fields, read without mutation.
        index
            Local turbine or target position.
        dimension
            Selected entity dimension, ``FC.TURBINE`` or ``FC.TARGET``.

        Returns
        -------
        samples
            Required fields as ``(state,)`` arrays. Target-point data selects the
            first ``FC.TPOINT`` entry.

        Raises
        ------
        KeyError
            If a required field is missing.
        ValueError
            If its dimensions do not describe scalar state/entity samples.
        """
        samples = {}
        for variable in self._farm_variables():
            if variable not in data:
                raise KeyError(f"{self.name}: Missing rose field '{variable}'")
            dims = data.dims[variable]
            values = np.broadcast_to(data[variable], tuple(data.sizes[d] for d in dims))
            if dims == (FC.STATE,):
                samples[variable] = values
            elif dims in ((FC.STATE, dimension), (FC.STATE, dimension, FC.TPOINT)):
                selected = values[:, index]
                samples[variable] = selected[:, 0] if selected.ndim == 2 else selected
            else:
                raise ValueError(
                    f"{self.name}: Invalid dimensions {dims} for rose field '{variable}'"
                )
        return samples

    def _point_sample(self, tdata: TData) -> dict[str, np.ndarray] | None:
        """Select the configured global point from its containing target tile.

        Parameters
        ----------
        tdata
            Supplied target fields with global ``FC.TARGET`` labels.

        Returns
        -------
        samples
            Selected scalar state fields, or ``None`` when the point is outside this tile.
        """
        labels = tdata[FC.TARGET]
        matches = np.flatnonzero(labels == self.point)
        if not len(matches):
            return None
        index = int(matches[0])
        return self._samples(tdata, index, FC.TARGET)

    def _edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct centered direction sectors and configured speed-bin edges.

        Returns
        -------
        directions
            Direction edges in degrees, centered on north for the first sector.
        speeds
            Speed edges, with an infinite final edge for frequency outputs using ``add_inf``.
        """
        delta = 360 / self.wd_sectors
        speeds = tuple(self.ws_bins)
        if self._contraction() == "frequency" and self.add_inf:
            speeds += (np.inf,)
        return np.arange(-delta / 2, 360, delta), np.asarray(speeds)

    def _histogram(self, samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Accumulate additive bin numerators and global-mean denominators.

        Parameters
        ----------
        samples
            Selected speed, direction, weight and optional field arrays, all ``(state,)``.

        Returns
        -------
        histogram
            Direction-by-speed ``sum`` and optional ``weight`` arrays.
            No states are retained and no chunk-local normalization is performed.
        """
        direction, speed = self._edges()
        wd = samples[self.wd_var].copy()
        wd[wd >= 360 - 180 / self.wd_sectors] -= 360
        ws = samples[self.ws_var]
        contraction = self._contraction()
        weights = samples[FV.WEIGHT]
        if contraction == "frequency":
            return {
                "sum": np.histogram2d(
                    wd, ws, (direction, speed), weights=100 * weights
                )[0]
            }
        variable = samples[self.variable]
        factor = weights if contraction == "weights" else np.ones_like(weights)
        result = {
            "sum": np.histogram2d(
                wd, ws, (direction, speed), weights=variable * factor
            )[0]
        }
        if contraction != "sum_no_weights":
            result["weight"] = np.histogram2d(
                wd, ws, (direction, speed), weights=factor
            )[0]
        return result

    def _sum_histograms(
        self, parts: list[dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """Sum compatible histogram components without normalizing local bins.

        Parameters
        ----------
        parts
            Nonempty sequence of histograms with identical component names and bin shapes.

        Returns
        -------
        histogram
            Independent arrays containing the elementwise sum of each component.

        Raises
        ------
        ValueError
            If no histogram parts are supplied.
        """
        if not parts:
            raise ValueError(f"{self.name}: No selected rose data")
        return {key: np.sum([part[key] for part in parts], axis=0) for key in parts[0]}

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Merge accumulators and normalize or render the global histogram.

        Parameters
        ----------
        algo
            Executing algorithm.
        n_chunks_states
            Total engine state chunks, each requiring the selected sample.
        n_chunks_targets
            Total engine target chunks.
        output_results
            Run-local buffers, consumed and replaced with a bin dataset or
            ``(axes, dataset)``. Denominator thresholds apply after global merging.

        Raises
        ------
        ValueError
            If the selected turbine or point is unavailable.
        """
        from xarray import Dataset

        chunks = output_results.pop(self.var("data"), {})
        if self.use_points and self.point < 0 and chunks:
            selected = max(c["last_target"] for c in chunks.values()) + 1 + self.point
            chunks = {
                key: chunk["candidates"][selected]
                for key, chunk in chunks.items()
                if selected in chunk["candidates"]
            }
        if {s for s, _ in chunks} != set(range(n_chunks_states)):
            raise ValueError(f"{self.name}: Selected rose point/turbine is unavailable")
        total = self._sum_histograms([chunks[key] for key in sorted(chunks)])
        direction, speed = self._edges()
        values = total["sum"]
        if "weight" in total:
            threshold = 1e-13 if self._contraction() == "weights" else 1
            denominator = np.where(total["weight"] < threshold, np.nan, total["weight"])
            values = values / denominator
        frequency = self._contraction() == "frequency"
        data_vars: dict[str, Any] = {
            "frequency" if frequency else self.variable: (
                (self.wd_var, self.ws_var),
                values,
            )
        }
        if frequency:
            for variable, bins in ((self.wd_var, direction), (self.ws_var, speed)):
                data_vars[f"bin_min_{variable}"] = (variable, bins[:-1])
                data_vars[f"bin_max_{variable}"] = (variable, bins[1:])
        data = Dataset(
            data_vars=data_vars,
            coords={
                self.wd_var: np.arange(0, 360, 360 / self.wd_sectors)
                if frequency
                else 0.5 * (direction[:-1] + direction[1:]),
                self.ws_var: 0.5 * (speed[:-1] + speed[1:]),
            },
            attrs={f"{self.wd_var}_bounds": direction, f"{self.ws_var}_bounds": speed},
        )
        power_unit = (
            _power_unit(algo)
            if not frequency
            and FV.amb2var.get(self.variable, self.variable) in (FV.P, FV.CAP, FV.MAX_P)
            else None
        )
        _set_units(data, power_unit)
        output_results[self.result_name] = self.process(algo, data)
