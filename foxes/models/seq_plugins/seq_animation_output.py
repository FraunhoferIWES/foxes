"""Adapt model-book flow animations to the Sequential iterator lifecycle."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from foxes.algorithms.sequential import SequentialPlugin
import foxes.constants as FC
import foxes.variables as FV

from .seq_wake_debug_plugin import SeqWakeDebugPlugin
from foxes.models.outputs import FlowAnimation, PointResultsOutput

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from xarray import Dataset
    from foxes.algorithms import Sequential


class SeqAnimationOutputPlugin(SequentialPlugin):
    """Collect sequential flow data and render selected animation output models.

    Numerical point evaluation happens during each iterator step, while dynamic
    wake history is available. Model-book animation renderers write files only
    after the iterator finishes. This adapter does not change Sequential's
    physical-variable ``outputs`` argument or its one-state-at-a-time contract.
    Numerical buffers belong to the plugin, never to the output model.
    An incomplete iteration does not write animation files.
    """

    def __init__(
        self,
        outputs: list[str | FlowAnimation],
        debug: bool = False,
    ) -> None:
        """Select animation models and optional wake-debug overlays.

        Parameters
        ----------
        outputs
            Animation model-book names or ``FlowAnimation`` instances, including
            ``FlowTurbineAnimation``. Other output types are rejected.
        debug
            Include the existing ``SeqWakeDebugPlugin`` wake-point/vector overlay.
        """
        super().__init__()
        self.outputs = outputs
        self.debug = debug

    def initialize(self, algo: Sequential) -> None:
        """Prepare animation grids, selections and per-run numerical buffers.

        Parameters
        ----------
        algo
            Initialized sequential iterator, before its first state.

        Raises
        ------
        KeyError
            If an output model-book name is unavailable.
        TypeError
            If a selected model is not a flow animation.
        ValueError
            If a selected animation does not supply calculation points.

        Notes
        -----
        Grids are retrieved from initialized output models without storing
        runtime arrays on those models. Selections and numerical buffers belong
        to this plugin for the duration of one iterator run.
        """
        # Resolve against the caller's model book, retaining no rendered frames.
        super().initialize(algo)
        self._runs: list[tuple[FlowAnimation, np.ndarray, Any, dict[Any, Any]]] = []
        for selected in self.outputs:
            output = (
                algo.mbook.outputs[selected] if isinstance(selected, str) else selected
            )
            if not isinstance(output, FlowAnimation):
                raise TypeError(
                    f"Sequential animation requires FlowAnimation, got {type(output).__name__}"
                )
            algo.init_model(output)
            loaded = algo.loaded_data
            selection = (
                None
                if output._selection is None
                else loaded["data_vars"][output._selection][1]
            )
            points = output.get_calc_points(algo)
            if points is None:
                raise ValueError(
                    f"{output.name}: Animation requires calculation points"
                )
            self._runs.append((output, points, selection, {}))
        self._debug = SeqWakeDebugPlugin() if self.debug and self._runs else None
        if self._debug is not None:
            self._debug.initialize(algo)

    def update(
        self, algo: Sequential, fres: Dataset, pres: Dataset | None = None
    ) -> None:
        """Evaluate selected grids while the current wake history is available.

        Parameters
        ----------
        algo
            Iterator at its current state.
        fres
            Current farm results on ``(FC.STATE, FC.TURBINE)`` in original
            turbine order, with exactly one state.
        pres
            Current iterator point results on ``(FC.STATE, FC.POINT)``, if
            available. Animation grids are evaluated separately.

        Notes
        -----
        Selected state-step payloads copy farm fields, state labels and
        ``(FC.STATE, FC.POINT)`` point values into plugin-owned buffers.
        Ordinary field names are selected explicitly through ``PointResultsOutput``;
        the output's ``ambient`` option controls whether wake effects are run.
        Running animation model configuration is not mutated.
        """
        # Use the normal point path so engine point chunk limits still apply.
        super().update(algo, fres, pres)
        assert algo.counter is not None
        for output, points, selection, chunks in self._runs:
            index = algo.counter
            if not any(
                c is None or c == index or c == (index, 0) for c in output.chunks
            ):
                continue
            if selection is not None and not selection[index]:
                continue
            variables = list(dict.fromkeys([*output._variables, FV.WD, FV.WS]))
            data = algo.calc_points(
                fres,
                points,
                outputs=PointResultsOutput(variables=variables),
                ambient=output.ambient,
            )
            chunks[(index, 0)] = dict(
                states=fres[FC.STATE].to_numpy().copy(),
                farm={
                    v: d.to_numpy().copy()
                    for v, d in fres.data_vars.items()
                    if d.dims == (FC.STATE, FC.TURBINE)
                },
                points=points[None, :, :],
                values={v: data[v].to_numpy().copy() for v in variables},
            )
        if self._debug is not None:
            self._debug.update(algo, fres, pres)

    def finalize(self, algo: Sequential) -> None:
        """Render state-ordered animations once and release numerical buffers.

        Parameters
        ----------
        algo
            Iterator after all states have completed.

        Raises
        ------
        ValueError
            If an explicitly selected state-step chunk does not exist.
        """
        # Validate step indices even when frame selection produces no payloads.
        super().finalize(algo)
        for output, _, _, chunks in self._runs:
            for selection in output.chunks:
                if selection is None:
                    continue
                indices = selection if isinstance(selection, tuple) else (selection, 0)
                if not (
                    indices[0] < len(algo.farm_results[FC.STATE]) and indices[1] == 0
                ):
                    raise ValueError(
                        f"{output.name}: Selected chunk {selection!r} is unavailable"
                    )
            if not chunks:
                continue
            generators: tuple[Callable[[Axes], Iterator[Any]], ...] = ()
            if self._debug is not None:
                # Each variable needs a fresh, identically selected debug history.
                debug = deepcopy(self._debug, {id(algo): algo})
                debug._data = [debug._data[i] for i, _ in sorted(chunks)]

                def create_debug_images(ax: Axes) -> Iterator[Any]:
                    return deepcopy(debug, {id(algo): algo}).gen_images(ax)

                generators = (create_debug_images,)
            output._render(algo, output._combine_chunks(chunks), generators)
            chunks.clear()
