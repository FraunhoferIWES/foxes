"""Bounded output point evaluation on private scientific chunk containers."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import TData

if TYPE_CHECKING:
    from xarray import Dataset

    from foxes.core import Algorithm, FData, MData, PointDataModelList


def farm_bounds(algo: Algorithm, output_name: str) -> Dataset:
    """Collect every turbine position and hub height needed to bound a grid.

    Parameters
    ----------
    algo
        Initialized algorithm supplying turbine types and farm geometry.
    output_name
        Model name included in contextual configuration errors.

    Returns
    -------
    bounds
        Dataset of ``FV.X``, ``FV.Y`` and ``FV.H`` on ``(state, turbine)``
        axes, broadcasting static positions across any state-dependent layouts.
        It is initialization-local and must not be stored on an output model.

    Raises
    ------
    ValueError
        If the farm has no initialized turbines or any hub height is unavailable.
    """
    from xarray import Dataset

    types = algo.farm_controller.turbine_types
    if types is None or not algo.farm.turbines:
        raise ValueError(
            f"{output_name}: Grid generation requires initialized turbines"
        )
    n_positions = max(t.xy.reshape(-1, 2).shape[0] for t in algo.farm.turbines)
    xy = np.stack(
        [
            np.broadcast_to(t.xy.reshape(-1, 2), (n_positions, 2))
            for t in algo.farm.turbines
        ],
        axis=1,
    )
    heights = [
        t.H if t.H is not None else types[i].H for i, t in enumerate(algo.farm.turbines)
    ]
    if any(h is None for h in heights):
        raise ValueError(f"{output_name}: Grid generation requires turbine hub heights")
    return Dataset(
        {
            FV.X: ((FC.STATE, FC.TURBINE), xy[:, :, 0]),
            FV.Y: ((FC.STATE, FC.TURBINE), xy[:, :, 1]),
            FV.H: (
                (FC.STATE, FC.TURBINE),
                np.broadcast_to(np.asarray(heights, dtype=float), xy.shape[:2]),
            ),
        }
    )


def join_point_tiles(tiles: list[dict[str, Any]], output_name: str) -> dict[str, Any]:
    """Join target-ordered tiles belonging to one state slab.

    Parameters
    ----------
    tiles
        Nonempty target-ordered payloads with matching state coordinates.
        Static ``points`` may have a singleton state axis.
    output_name
        Model name included in contextual state-mismatch errors.

    Returns
    -------
    slab
        Combined fields and ``(state, point, 3)`` coordinates. Farm overlays
        and state labels are borrowed from the first tile; point/field arrays
        are concatenated without sorting user-supplied target positions.

    Raises
    ------
    ValueError
        If target tiles disagree on their state coordinates.
    """
    if any(not np.array_equal(t["states"], tiles[0]["states"]) for t in tiles):
        raise ValueError(f"{output_name}: Target tiles have inconsistent states")
    return dict(
        states=tiles[0]["states"],
        farm=tiles[0]["farm"],
        points=np.concatenate(
            [
                np.broadcast_to(t["points"], (len(t["states"]), *t["points"].shape[1:]))
                for t in tiles
            ],
            axis=1,
        ),
        values={
            v: np.concatenate([t["values"][v] for t in tiles], axis=1)
            for v in tiles[0]["values"]
        },
    )


def point_blocks(
    algo: Algorithm,
    mdata: MData,
    fdata: FData,
    points: np.ndarray,
    models: PointDataModelList,
    parameters: list[dict[str, Any]],
    chunk_size: int | None,
) -> Iterator[TData]:
    """Evaluate bounded point slices on private, downwind-ordered chunk copies.

    Parameters
    ----------
    algo
        Running algorithm supplying wake-model context.
    mdata
        Farm-ordered chunk metadata; shared extra data is borrowed read-only.
    fdata
        Farm-ordered scientific inputs, copied before downstream reordering.
    points
        Static ``(point, 3)`` or full-input ``(state, point, 3)`` coordinates.
        State-dependent points are selected using the chunk's global offset.
    models
        Initialized point models, treated as read-only throughout calculation.
    parameters
        Calculation parameters in point-model order.
    chunk_size
        Maximum targets per slice; ``None`` evaluates all requested points.

    Yields
    ------
    target_data
        Calculated private target container with fields ordered as
        ``(FC.STATE, FC.TARGET, FC.TPOINT)``. No input arrays are modified.

    Raises
    ------
    ValueError
        If points are empty, the chunk limit is nonpositive, or state-dependent
        coordinates cannot be selected without a state-chunk offset.
    """
    # Wake evaluation expects downwind order, whereas output inputs use farm order.
    md = deepcopy(mdata, {id(mdata.extra_data): mdata.extra_data})
    fd = deepcopy(fdata)
    order = fd[FV.ORDER]
    for data in (md, fd):
        for variable, dims in data.dims.items():
            if dims[:2] == (FC.STATE, FC.TURBINE) and variable != FV.ORDER:
                values = data[variable]
                if values.shape[1] > 1:
                    data[variable] = values[np.arange(len(order))[:, None], order]

    # Select only the active states when coordinates depend on the input state.
    if points.ndim == 3:
        start = mdata.states_i0(counter=True)
        if start is None or md.n_states is None:
            raise ValueError(
                "State-dependent output points require state chunk offsets"
            )
        points = points[start : start + md.n_states]
    n_points = points.shape[-2]
    if n_points == 0:
        raise ValueError("Output calculation points must not be empty")
    size = n_points if chunk_size is None else chunk_size
    if size < 1:
        raise ValueError("Output point chunk size must be positive")
    for start in range(0, n_points, size):
        block = points[..., start : start + size, :]
        targets = TData.from_points(
            np.broadcast_to(block, (md.n_states, *block.shape[-2:])), mdata=md
        )
        targets.update(models.calculate(algo, md, fd, targets, parameters=parameters))
        yield targets
