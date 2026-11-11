"""Constructor-defined output presets and parameterized factories."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import foxes.models as fm
import foxes.variables as FV
from foxes.utils import FDict

if TYPE_CHECKING:
    from foxes.core import Output

    from .model_book import ModelBook


def _populate_outputs(book: ModelBook) -> None:
    """Populate constructor-defined collectors, writers and plotting outputs.

    Parameters
    ----------
    book
        Model book receiving fresh output instances and factory configuration.
    """
    book.outputs = FDict(_name="outputs")
    _populate_plot_outputs(book)
    _populate_slice_outputs(book)
    _populate_animation_outputs(book)
    _populate_result_outputs(
        book,
        "farm_results",
        _farm_variable_sets(),
        (
            fm.outputs.FarmResultsOutput,
            fm.outputs.FarmResultsCountWriter,
            fm.outputs.FarmResultsChunkWriter,
            fm.outputs.FarmResultsPeriodWriter,
        ),
    )
    _populate_result_outputs(
        book,
        "point_results",
        _point_variable_sets(),
        (
            fm.outputs.PointResultsOutput,
            fm.outputs.PointResultsCountWriter,
            fm.outputs.PointResultsChunkWriter,
            fm.outputs.PointResultsPeriodWriter,
        ),
    )


def _populate_plot_outputs(book: ModelBook) -> None:
    """Populate interactive layout and rose presets.

    Parameters
    ----------
    book
        Model book whose output registry is extended in place.
    """
    book.outputs["farm_layout"] = fm.outputs.FarmLayoutOutput(show=True)
    book.outputs["rose_plot"] = fm.outputs.RosePlotOutput(show=True)
    book.outputs["states_rose_plot"] = fm.outputs.StatesRosePlotOutput(
        point=(0.0, 0.0, 100.0), show=True
    )
    book.outputs["wind_rose_bin_plot"] = fm.outputs.WindRoseBinPlot(
        variable=FV.P, ws_bins=(0, 5, 10, 15, 20, 25), show=True
    )
    book.outputs["state_turbine_map"] = fm.outputs.StateTurbineMap(show=True)
    book.outputs["state_turbine_table"] = fm.outputs.StateTurbineTable()
    book.outputs["point_calculator"] = fm.outputs.PointCalculator()
    book.outputs["rotor_point_plot"] = fm.outputs.RotorPointPlot(show=True)
    book.outputs["turbine_type_curves"] = fm.outputs.TurbineTypeCurves(show=True)


def _populate_slice_outputs(book: ModelBook) -> None:
    """Populate plane/reduction slice and static-flow factories.

    Parameters
    ----------
    book
        Model book whose output registry is extended in place.
    """
    slice_outputs: tuple[
        tuple[str, type[fm.outputs.SliceData], dict[str, Any]], ...
    ] = (
        ("slice_data", fm.outputs.SliceData, {}),
        ("slices_data", fm.outputs.SlicesData, {"positions": (100.0,)}),
        ("flow_plot", fm.outputs.FlowPlots2D, {"show": True}),
    )
    for name, output_type, options in slice_outputs:
        book.outputs[name] = output_type(**options)
        book.outputs.add_factory(
            output_type,
            f"{name}_<plane>_<reduction>",
            plane=["xy", "xz", "yz"],
            reduction=["states", "mean"],
            kwargs=options,
        )


def _populate_animation_outputs(book: ModelBook) -> None:
    """Populate noninteractive animation writers and chunk factories.

    Parameters
    ----------
    book
        Model book whose output registry is extended in place.
    """
    book.outputs["create_gif"] = fm.outputs.FlowAnimation(
        base_name="ani.gif", show=False
    )
    book.outputs["create_mp4"] = fm.outputs.FlowAnimation(
        base_name="ani.mp4", show=False
    )
    book.outputs["create_turbine_gif"] = fm.outputs.FlowTurbineAnimation(
        base_name="ani.gif", show=False
    )
    book.outputs["create_turbine_mp4"] = fm.outputs.FlowTurbineAnimation(
        base_name="ani.mp4", show=False
    )
    book.outputs.add_factory(
        fm.outputs.FlowAnimation,
        "chunk_gif_<chunk>",
        chunk=lambda c: int(c),
        hints={"chunk": "(Chunk index)"},
        kwargs=dict(base_name="ani.gif", show=False),
    )
    book.outputs.add_factory(
        fm.outputs.FlowAnimation,
        "chunk_mp4_<chunk>",
        chunk=lambda c: int(c),
        hints={"chunk": "(Chunk index)"},
        kwargs=dict(base_name="ani.mp4", show=False),
    )


def _farm_variable_sets() -> dict[str, list[str] | None]:
    """Create independent named farm-variable selections.

    Returns
    -------
    selections
        Variable lists for output factories; ``all`` selects every field.
    """
    return {
        "all": None,
        "default": [
            FV.X,
            FV.Y,
            FV.H,
            FV.D,
            FV.WD,
            FV.YAW,
            FV.RHO,
            FV.AMB_TI,
            FV.TI,
            FV.AMB_REWS,
            FV.REWS,
            FV.CT,
            FV.AMB_P,
            FV.P,
            FV.WEIGHT,
        ],
        "P": [
            FV.AMB_P,
            FV.P,
            FV.WEIGHT,
        ],
        "wsP": [
            FV.AMB_REWS,
            FV.REWS,
            FV.AMB_P,
            FV.P,
            FV.WEIGHT,
        ],
        "wstiP": [
            FV.AMB_REWS,
            FV.REWS,
            FV.AMB_TI,
            FV.TI,
            FV.AMB_P,
            FV.P,
            FV.WEIGHT,
        ],
        "rhowstiP": [
            FV.AMB_RHO,
            FV.AMB_REWS,
            FV.REWS,
            FV.AMB_TI,
            FV.TI,
            FV.AMB_P,
            FV.P,
            FV.WEIGHT,
        ],
        "analysis": [
            FV.CAP,
            FV.AMB_CAPF,
            FV.CAPF,
            FV.EFF,
            FV.WEIGHT,
        ],
    }


def _point_variable_sets() -> dict[str, list[str] | None]:
    """Create independent point-variable selections for collectors and writers.

    Returns
    -------
    selections
        Standard ambient/waked wind fields and weights under ``default``;
        ``all`` retains every available point field.
    """
    return {
        "all": None,
        "default": [
            FV.WD,
            FV.WS,
            FV.TI,
            FV.RHO,
            FV.AMB_WD,
            FV.AMB_WS,
            FV.AMB_TI,
            FV.AMB_RHO,
            FV.WEIGHT,
        ],
    }


def _populate_result_outputs(
    book: ModelBook,
    prefix: str,
    variable_sets: dict[str, list[str] | None],
    classes: tuple[type[Output], type[Output], type[Output], type[Output]],
) -> None:
    """Populate analogous farm or point collectors and NetCDF writer selectors.

    Parameters
    ----------
    book
        Model book whose output registry is extended in place.
    prefix
        Ungrouped ``farm_results`` or ``point_results`` family. Turbine
        aggregation selects ``farms_results``, ``cluster_results`` or
        ``total_results``.
    variable_sets
        Independent named variable selections including ``default`` and ``all``.
    classes
        Collector, count writer, chunk writer and calendar-period writer classes.
    """
    spatial_parameter = (
        "aggregate_turbines" if prefix == "farm_results" else "aggregate_points"
    )
    aggregates = (
        ("wind_farm", "cluster", "all", None)
        if prefix == "farm_results"
        else ("all", None)
    )
    for filtering, ambient, waked in (
        ("_amb", True, False),
        ("_waked", False, True),
        ("", True, True),
    ):
        for aggregate in aggregates:
            if aggregate == "wind_farm":
                variant_prefix = "farms_results"
            elif aggregate == "cluster":
                variant_prefix = "cluster_results"
            elif aggregate == "all" and prefix == "farm_results":
                variant_prefix = "total_results"
            else:
                variant_prefix = prefix
            for states in (True, False):
                configuration = {
                    spatial_parameter: aggregate,
                    "aggregate_states": states,
                    "ambient_vars": ambient,
                    "waked_vars": waked,
                }
                for fields in (False, True):
                    stem = (
                        variant_prefix + filtering + ("_<variables>" if fields else "")
                    )
                    if aggregate == "all" and prefix == "point_results":
                        stem += "_aggregate_all"
                    if states:
                        stem += "_mean"
                    _populate_result_variant(
                        book, stem, variable_sets, classes, configuration
                    )


def _writer_fields(
    variable_sets: dict[str, list[str] | None], stem: str, selection: str
) -> dict[str, Any]:
    """Select fields and keep different named writer configurations in distinct files."""
    return {
        "variables": variable_sets[selection],
        "base_name": stem.replace("<variables>", selection),
    }


def _populate_result_variant(
    book: ModelBook,
    stem: str,
    variable_sets: dict[str, list[str] | None],
    classes: tuple[type[Output], type[Output], type[Output], type[Output]],
    configuration: dict[str, Any],
) -> None:
    """Register every writer policy and collector for one spatial/state/filter variant."""
    collector, count, chunks, period = classes
    for output_type, suffix in (
        (count, "_write_n<n_states_per_file>"),
        (chunks, "_write_chunks"),
        (period, "_write_<period>"),
        (collector, ""),
    ):
        options: dict[str, Any] = {}
        arguments: dict[str, str | None] = {}
        kwargs = dict(configuration)
        if "<variables>" in stem:
            options["variables"] = (
                variable_sets
                if output_type is collector
                else partial(_writer_fields, variable_sets, stem)
            )
            arguments["variables"] = "variables" if output_type is collector else None
        else:
            kwargs["variables"] = (
                None if output_type is count else variable_sets["default"]
            )
            if output_type is not collector:
                kwargs["base_name"] = stem
        if output_type is count:
            options["n_states_per_file"] = int
        elif output_type is period:
            options["period"] = ("day", "week", "month", "year")
        name = stem + suffix
        if "<" in name:
            book.outputs.add_factory(
                output_type, name, kwargs=kwargs, var2arg=arguments, **options
            )
        else:
            book.outputs[name] = output_type(**kwargs)
