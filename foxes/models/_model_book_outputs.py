"""Constructor-defined output presets and parameterized factories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import foxes.models as fm
import foxes.variables as FV
from foxes.utils import FDict

if TYPE_CHECKING:
    from .model_book import ModelBook


def _populate_outputs(book: ModelBook) -> None:
    """Populate constructor-defined collectors, writers and plotting outputs.

    Parameters
    ----------
    book
        Model book receiving fresh output instances and factory configuration.
    """
    book.outputs = FDict(_name="outputs")
    book.outputs["point_results"] = fm.outputs.PointResultsOutput()
    _populate_plot_outputs(book)
    _populate_slice_outputs(book)
    _populate_animation_outputs(book)
    _populate_farm_result_outputs(book)


def _populate_plot_outputs(book: ModelBook) -> None:
    """Populate interactive layout, evaluation and rose presets.

    Parameters
    ----------
    book
        Model book whose output registry is extended in place.
    """
    book.outputs["farm_layout"] = fm.outputs.FarmLayoutOutput(show=True)
    book.outputs["farm_results_eval"] = fm.outputs.FarmResultsEval()
    book.outputs["rose_plot"] = fm.outputs.RosePlotOutput(show=True)
    book.outputs["states_rose_plot"] = fm.outputs.StatesRosePlotOutput(
        point=(0.0, 0.0, 100.0), show=True
    )
    book.outputs["wind_rose_bin_plot"] = fm.outputs.WindRoseBinPlot(
        variable=FV.P, ws_bins=(0, 5, 10, 15, 20, 25), show=True
    )


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
        ]
    }


def _populate_farm_result_outputs(book: ModelBook) -> None:
    """Populate farm collectors and bounded NetCDF writers.

    Parameters
    ----------
    book
        Model book whose output registry is extended in place.
    """
    farm_outputs = _farm_variable_sets()
    book.outputs["farm_results"] = fm.outputs.FarmResultsOutput(
        ambient_vars=True,
        waked_vars=True,
        variables=farm_outputs["default"],
    )
    book.outputs["farm_results_amb"] = fm.outputs.FarmResultsOutput(
        ambient_vars=True,
        waked_vars=False,
        variables=farm_outputs["default"],
    )
    book.outputs.add_factory(
        fm.outputs.FarmResultsOutput,
        "farm_results_amb_<variables>",
        variables=lambda v: farm_outputs[v],
        hints={"variables": f"(Variable set: {', '.join(farm_outputs.keys())})"},
        kwargs=dict(ambient_vars=True, waked_vars=False),
    )
    book.outputs.add_factory(
        fm.outputs.FarmResultsOutput,
        "farm_results_waked_<variables>",
        variables=lambda v: farm_outputs[v],
        hints={"variables": f"(Variable set: {', '.join(farm_outputs.keys())})"},
        kwargs=dict(ambient_vars=False, waked_vars=True),
    )
    book.outputs.add_factory(
        fm.outputs.FarmResultsCountWriter,
        "farm_results_write_n<n_states_per_file>",
        n_states_per_file=lambda n: int(n),
        hints={"states_per_file": "(Number of states per file)"},
        kwargs=dict(base_name="farm_results", ambient_vars=True, waked_vars=True),
    )
    book.outputs.add_factory(
        fm.outputs.FarmResultsOutput,
        "farm_results_<variables>",
        variables=lambda v: farm_outputs[v],
        hints={"variables": f"(Variable set: {', '.join(farm_outputs.keys())})"},
        kwargs=dict(ambient_vars=True, waked_vars=True),
    )
    book.outputs["farm_results_write_chunks"] = fm.outputs.FarmResultsChunkWriter(
        base_name="farm_results",
        ambient_vars=True,
        waked_vars=True,
        variables=farm_outputs["default"],
    )
    book.outputs["farm_results_write_day"] = fm.outputs.FarmResultsPeriodWriter(
        base_name="farm_results",
        period="day",
        ambient_vars=True,
        waked_vars=True,
        variables=farm_outputs["default"],
    )
    book.outputs["farm_results_write_week"] = fm.outputs.FarmResultsPeriodWriter(
        base_name="farm_results",
        period="week",
        ambient_vars=True,
        waked_vars=True,
        variables=farm_outputs["default"],
    )
    book.outputs["farm_results_write_month"] = fm.outputs.FarmResultsPeriodWriter(
        base_name="farm_results",
        period="month",
        ambient_vars=True,
        waked_vars=True,
        variables=farm_outputs["default"],
    )
    book.outputs["farm_results_write_year"] = fm.outputs.FarmResultsPeriodWriter(
        base_name="farm_results",
        period="year",
        ambient_vars=True,
        waked_vars=True,
        variables=farm_outputs["default"],
    )
