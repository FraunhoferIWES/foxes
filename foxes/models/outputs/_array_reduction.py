"""Mergeable NumPy statistics for output-worker dimension reductions."""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class ArrayStats(TypedDict, total=False):
    """Mergeable statistics with no retained reduction axis.

    ``value`` contains a reduced sum/extremum, except for standard deviation,
    where it contains the local mean. Means additionally retain valid ``count``;
    deviations retain both ``count`` and the centered squared-deviation sum
    ``m2``. These mappings belong to worker results, never to a running model.
    """

    value: np.ndarray
    count: np.ndarray
    m2: np.ndarray


def reduce_array(
    values: np.ndarray,
    operation: str | None,
    axis: int = 0,
    weights: np.ndarray | None = None,
) -> ArrayStats:
    """Reduce one axis into mergeable, NaN-aware statistics.

    Parameters
    ----------
    values
        Input array, read without mutation.
    operation
        ``"weights"``, ``"mean"``, ``"std"``, ``"sum"``, ``"min"`` or
        ``"max"``; ``None`` copies values without reducing their dimensions.
    axis
        Axis eliminated from every returned statistic.
    weights
        Weights broadcastable to ``values``, required for ``"weights"``.
        Weighted sums are not normalized. Callers must reject NaNs carrying
        nonzero weight before invoking this helper.

    Returns
    -------
    statistics
        Reduced ``value`` and, only when needed, ``count`` and ``m2`` arrays.
        Standard-deviation moments use the population convention.

    Raises
    ------
    ValueError
        If weighted reduction is requested without weights.
    KeyError
        If the operation is unknown.
    """
    if operation is None:
        return {"value": values.copy()}
    if operation == "weights":
        if weights is None:
            raise ValueError("Weighted output reduction requires state weights")
        return {
            "value": np.sum(
                np.where(np.isnan(values), 0.0, values) * weights, axis=axis
            )
        }
    if operation in ("mean", "std"):
        count = np.sum(~np.isnan(values), axis=axis)
        total = np.nansum(values, axis=axis)
        if operation == "mean":
            return {"value": total, "count": count}
        mean = np.divide(
            total, count, out=np.zeros_like(total, dtype=float), where=count != 0
        )
        stats: ArrayStats = {"value": mean, "count": count}
        delta = np.where(np.isnan(values), 0.0, values - np.expand_dims(mean, axis))
        stats["m2"] = np.sum(delta**2, axis=axis)
        return stats
    reducers = {"sum": np.nansum, "min": np.nanmin, "max": np.nanmax}
    if operation not in reducers:
        raise KeyError(f"Unknown output reduction '{operation}'")
    return {"value": np.asarray(reducers[operation](values, axis=axis))}


def merge_stats(parts: list[ArrayStats], operation: str | None) -> ArrayStats:
    """Merge worker statistics without restoring eliminated dimensions.

    Parameters
    ----------
    parts
        Nonempty sequence of compatible statistics for disjoint chunks.
    operation
        Reduction used to create every part. ``None`` returns the first part,
        appropriate for unchanged static values.

    Returns
    -------
    statistics
        Additive sums/counts, merged extrema or parallel centered moments.
        Inputs are not mutated; the ``None`` case returns the original mapping.

    Raises
    ------
    ValueError
        If no parts are supplied.
    KeyError
        If the operation is unknown.
    """
    if not parts:
        raise ValueError("Cannot merge an empty output reduction")
    if operation is None:
        return parts[0]
    if operation == "mean":
        return {
            "value": np.sum([p["value"] for p in parts], axis=0),
            "count": np.sum([p["count"] for p in parts], axis=0),
        }
    if operation == "std":
        result: ArrayStats = {
            "value": parts[0]["value"].copy(),
            "count": parts[0]["count"].copy(),
            "m2": parts[0]["m2"].copy(),
        }
        for part in parts[1:]:
            count = result["count"] + part["count"]
            delta = part["value"] - result["value"]
            fraction = np.divide(
                part["count"], count, out=np.zeros_like(delta), where=count != 0
            )
            result["m2"] = (
                result["m2"] + part["m2"] + delta**2 * result["count"] * fraction
            )
            result["value"] = result["value"] + delta * fraction
            result["count"] = count
        return result
    if operation in ("weights", "sum"):
        return {"value": np.sum([p["value"] for p in parts], axis=0)}
    return reduce_array(np.stack([p["value"] for p in parts]), operation)


def stats_value(stats: ArrayStats, operation: str | None) -> np.ndarray:
    """Finalize statistics into a reduced result array.

    Parameters
    ----------
    stats
        Statistics produced by ``reduce_array`` or ``merge_stats``.
    operation
        Original reduction; only means and standard deviations require
        normalization. Other operations return the stored ``value`` directly.

    Returns
    -------
    values
        Reduced values with NaN where a mean or deviation has no valid samples.
        Deviations use ``ddof=0``. The statistics are not mutated.
    """
    if operation not in ("mean", "std"):
        return stats["value"]
    count = stats["count"]
    if operation == "std":
        variance = np.divide(
            stats["m2"],
            count,
            out=np.full_like(stats["value"], np.nan),
            where=count != 0,
        )
        return np.sqrt(np.maximum(variance, 0.0))
    return np.divide(
        stats["value"],
        count,
        out=np.full_like(stats["value"], np.nan, dtype=float),
        where=count != 0,
    )
