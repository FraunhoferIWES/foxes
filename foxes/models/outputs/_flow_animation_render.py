"""Parent-side preparation, rendering and cleanup of planar flow animations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from cycler import cycler
from xarray import Dataset

import foxes.constants as FC

from . import _flow_figures
from ._variable_info import get_data_info
from .grids import np2np_sp, project_points

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from foxes.core import Algorithm
    from foxes.utils import Dict

    from .flow_animations import FlowAnimation


def turbine_frames(
    farm: Dataset, turbines: tuple[int, ...], variable: str, fig: Figure, ax: Axes
) -> Iterator[tuple[Figure, list[Line2D]]]:
    """Reveal turbine histories in global state order without reducing data.

    Parameters
    ----------
    farm
        Read-only farm fields on ``(FC.STATE, FC.TURBINE)``.
    turbines
        Original turbine indices in plotting order.
    variable
        Farm field to plot against positional state counts.
    fig
        Caller-owned animation figure.
    ax
        Trace axes, receiving lines and an upper-right legend.

    Yields
    ------
    frame
        Figure and line artists for each state. Frame ``si`` shows
        ``data[:si]``; the first history is empty. No figures are shown or saved.
    """
    ax.set_xlabel("State")
    ax.set_ylabel(variable)
    colors = cycler(color="bgrcmyk")
    values = farm[variable].to_numpy()
    labeled = set()
    for si in range(len(values)):
        artists = []
        ax.set_prop_cycle(colors)
        for ti in turbines:
            label = None if ti in labeled else f"Turbine {ti}"
            artists += ax.plot(range(si), values[:si, ti], label=label, animated=True)
            labeled.add(ti)
        ax.legend(loc="upper right")
        yield fig, artists


def _grid_data(
    output: FlowAnimation, payload: dict[str, Any]
) -> tuple[tuple[Any, ...], np.ndarray | None]:
    """Prepare a fixed rectangular grid without resampling supplied targets.

    Parameters
    ----------
    output
        Initialized animation supplying the plane and projection direction.
    payload
        Combined ``points``, ``states`` and ``values`` in supplied target order.
        Points have dimensions ``(state, point, 3)``.

    Returns
    -------
    grid
        Three projected coordinate entries and the first frame's world points.
    values
        Rectangular ``(state, a, b, variable)`` values, or ``None`` for moving,
        irregular or degenerate target coordinates. Inputs are not mutated.
    """
    points = payload["points"]
    projected = np.round(
        project_points(
            points, output.plane, output._grid_options.get("x_direction", 270)
        ),
        10,
    )
    ia, ib = ("xyz".index(axis) for axis in output.plane)
    ic = next(i for i in range(3) if i not in (ia, ib))
    x, y = np.unique(projected[0, :, ia]), np.unique(projected[0, :, ib])
    coordinates = [np.unique(projected[0, :, i]) for i in range(3)]
    grid = tuple(c[0] if i == ic else c for i, c in enumerate(coordinates)) + (
        points[0],
    )
    rectangular = (
        len(x) * len(y) == points.shape[1]
        and len(x) > 1
        and len(y) > 1
        and np.all(projected == projected[0])
        and len(coordinates[ic]) == 1
        and len(np.unique(projected[0][:, (ia, ib)], axis=0)) == points.shape[1]
    )
    if not rectangular:
        return grid, None
    order = np.lexsort((projected[0, :, ib], projected[0, :, ia]))
    gridded = np2np_sp(
        {v: d[:, order] for v, d in payload["values"].items()},
        payload["states"],
        x,
        y,
    )
    return grid, gridded


def _variable_options(
    output: FlowAnimation,
    variable: str,
    values: dict[str, np.ndarray],
    plot_pars: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Resolve one field's renderer parameters and output path.

    Parameters
    ----------
    output
        Initialized animation supplying filename and projection configuration.
    variable
        Scientific field rendered into this animation file.
    values
        Available field arrays, used only to preserve their variable ordering.
    plot_pars
        Parent-owned plot options; ``None`` selects rendering defaults.

    Returns
    -------
    parameters
        Field, color-scale and projection options shared by renderers.
    options
        Independent plot options; the output's configuration is not mutated.
    path
        Resolved filename with the variable appended to its stem.
    """
    options = dict(plot_pars or {})
    options.setdefault("vlabel", get_data_info(variable)[1])
    parameters = dict(
        var=variable,
        variables=list(values),
        data_format="numpy",
        vmin=options.pop("vmin", None),
        vmax=options.pop("vmax", None),
        x_direction=output._grid_options.get("x_direction", 270),
    )
    filename = output._base_name.with_name(
        f"{output._base_name.stem}_{variable}{output._base_name.suffix}"
    )
    return parameters, options, output.get_fpath(filename)


def _flow_generator(
    output: FlowAnimation,
    payload: dict[str, Any],
    variable: str,
    fig: Figure,
    ax: Axes,
    parameters: dict[str, Any],
    options: dict[str, Any],
) -> Iterator[Any]:
    """Select grid or scatter frames without modifying the scientific payload.

    Parameters
    ----------
    output
        Animation supplying the plane and scatter renderer.
    payload
        Parent-local render data with ``farm_dataset``, ``grid`` and optional
        ``gridded`` values, in addition to the collected scientific fields.
    variable
        Field to animate.
    fig
        Figure retained by the caller until display and encoding finish.
    ax
        Axes onto which all frame artists are drawn.
    parameters
        Field and projection settings forwarded to the renderer.
    options
        Independent plotting options that a renderer may consume.

    Returns
    -------
    frames
        Iterator yielding ``(figure, artists)`` pairs in global state order.
    """
    if payload["gridded"] is None:
        return output._scatter_frames(fig, ax, payload, variable, parameters, options)
    generator = getattr(_flow_figures, f"_gen_states_fig_{output.plane}")
    return generator(
        payload["farm_dataset"],
        output,
        (parameters, payload["gridded"], payload["states"], payload["grid"]),
        fig=fig,
        ax=ax,
        animated=True,
        ret_im=True,
        **options,
    )


def _render_variable(
    output: FlowAnimation,
    algo: Algorithm,
    payload: dict[str, Any],
    variable: str,
    output_results: Dict[str, Any],
    extra_generators: tuple[Callable[[Axes], Iterator[Any]], ...],
) -> None:
    """Save and optionally display one animation, closing its figure on failure.

    Parameters
    ----------
    output
        Initialized animation; no figures or arrays are retained on it.
    algo
        Algorithm supplying farm metadata and diagnostic verbosity.
    payload
        Parent-local prepared render data; scientific arrays remain read-only.
    variable
        Field whose name determines the file suffix.
    output_results
        Run-local results containing this animation's private rendering options.
    extra_generators
        Factories for synchronized artists on the shared flow axes.
    """
    import matplotlib.pyplot as plt

    parameters, options, path = _variable_options(
        output, variable, payload["values"], output_results[output.var("plot_pars")]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    verbosity = algo.verbosity if output.verbosity is None else output.verbosity
    if verbosity > 0:
        print(f"{output.name}: Creating animation {path}")
    n_frames = len(payload["states"])
    writer = "pillow" if path.suffix.lower() == ".gif" else "ffmpeg"
    if verbosity > 1:
        print(
            f"{output.name}: Preparing {n_frames} frames with "
            f"{payload['points'].shape[1]} points per frame"
        )
    fig, ax = output._create_figure()
    try:
        animator = output._create_animator(
            algo, payload["farm_dataset"], fig, ax, output_results
        )
        for create_generator in extra_generators:
            animator.add_generator(create_generator(ax))
        animator.add_generator(
            _flow_generator(output, payload, variable, fig, ax, parameters, options)
        )
        animation = animator.animate(
            verbosity=max(verbosity - 1, 0), interval=1000 / output.fps
        )
        if verbosity > 1:
            print(f"{output.name}: Writing {path}")
            print(
                f"{output.name}: Encoding {n_frames} frames at "
                f"{output.fps:g} fps with {writer}"
            )
        animation.save(
            filename=path,
            writer=writer,
            fps=output.fps,
            progress_callback=(
                lambda frame, total: print(
                    f"{output.name}: Encoding frame {frame + 1}/{total}"
                )
            )
            if verbosity > 2
            else None,
        )
        if verbosity > 1:
            print(f"{output.name}: Finished writing {path}")
        output._show_figures()
    finally:
        plt.close(fig)


def render_flow_animation(
    output: FlowAnimation,
    algo: Algorithm,
    payload: dict[str, Any],
    output_results: Dict[str, Any],
    extra_generators: tuple[Callable[[Axes], Iterator[Any]], ...] = (),
) -> None:
    """Render collected flow fields exclusively in parent-side finalization.

    Parameters
    ----------
    output
        Initialized animation supplying frozen constructor configuration.
        ``nofig`` suppresses preparation, encoding and display.
    algo
        Algorithm supplying farm metadata and diagnostic verbosity.
    payload
        Combined state-ordered field arrays, coordinates and overlay geometry.
        Arrays are borrowed, not copied or retained on the model.
    output_results
        Run-local results containing this animation's private rendering options.
    extra_generators
        Factories adding synchronized artists to every variable's animation.

    Notes
    -----
    Files are saved before blocking display. The animation stays alive until
    display returns, and its figure is closed even when encoding or display
    raises an exception.
    """
    if output.nofig:
        return
    farm = Dataset(
        {v: ((FC.STATE, FC.TURBINE), d) for v, d in payload["farm"].items()},
        coords={FC.STATE: payload["states"]},
    )
    grid, gridded = _grid_data(output, payload)
    prepared = {
        **payload,
        "farm_dataset": farm,
        "grid": grid,
        "gridded": gridded,
    }
    for variable in output._variables:
        _render_variable(
            output, algo, prepared, variable, output_results, extra_generators
        )
