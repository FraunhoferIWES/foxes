from __future__ import annotations

from xarray import Dataset
from typing import Any, TYPE_CHECKING, cast
import matplotlib.pyplot as plt

from foxes.config import get_output_path
import foxes.constants as FC
import foxes.variables as FV

from .flow_plots import FlowPlots2D
from ..animation import Animator

if TYPE_CHECKING:
    from foxes.core import Algorithm, FData, MData, TData


def write_chunk_ani_xy(
    algo: Algorithm,
    mdata: MData,
    fdata: FData,
    tdata: TData | None = None,
    vars: list[str] = [FV.WS],
    resolution: float = 100.0,
    figsize: tuple[int, int] = (8, 8),
    fpath_base: str = "chunk_animation.gif",
    n_img_points: tuple[int, int] | None = None,
    xmin: float | None = None,
    ymin: float | None = None,
    xmax: float | None = None,
    ymax: float | None = None,
    z: float | None = None,
    xspace: float = 500.0,
    yspace: float = 500.0,
    states_sel: Any = None,
    states_isel: Any = None,
    fps: int = 4,
    **kwargs: Any,
) -> None:
    """
    Writes an animation of a chunk calculation to file.

    Parameters
    ----------
    algo
        The calculation algorithm
    mdata
        The model data
    fdata
        The farm data
    tdata
        The point data, for point calculations
    vars
        The variables to be plotted
    resolution
        The resolution of the plot
    figsize
        The figure size
    fpath_base
        The base name for the output files, including suffix,
        e.g. 'output/chunk_ani.gif' or 'output/chunk_ani.mp4'
    n_img_points
        The number of image points, or `None` for automatic
    xmin
        The minimum x coordinate, or `None` for automatic
    ymin
        The minimum y coordinate, or `None` for automatic
    xmax
        The maximum x coordinate, or `None` for automatic
    ymax
        The maximum y coordinate, or `None` for automatic
    z
        The z coordinate of the slice, or `None` for automatic
    xspace
        The spacing in x direction if xmin/xmax are automatic
    yspace
        The spacing in y direction if ymin/ymax are automatic
    states_sel
        Reduce to selected states
    states_isel
        Reduce to the selected states indices
    fps
        The frames per second for the animation
    kwargs
        Additional keyword arguments for the plotting function

    """
    # case calc_farm:
    if mdata is not None and fdata is not None and tdata is None:
        try:
            if states_isel is not None:
                mdata = cast(MData, mdata.get_slice(FC.STATE, states_isel, force=True))
                fdata = cast(FData, fdata.get_slice(FC.STATE, states_isel, force=True))
            if states_sel is not None:
                n_states = mdata.n_states
                assert n_states is not None
                s = [i for i in range(n_states) if mdata[FC.STATE][i] in states_sel]
                mdata = cast(MData, mdata.get_slice(FC.STATE, s, force=True))
                fdata = cast(FData, fdata.get_slice(FC.STATE, s, force=True))
        except IndexError:
            return

        farm_results = Dataset(
            data_vars={
                v: ((FC.STATE, FC.TURBINE), d)
                for v, d in fdata.items()
                if d.shape == (fdata.n_states, fdata.n_turbines)
            },
            coords={FC.STATE: fdata[FC.STATE]},
        )

        fpath = get_output_path(fpath_base)
        odir = fpath.parent
        odir.mkdir(parents=True, exist_ok=True)
        base_name = fpath.stem
        suffix = fpath.suffix

        for var in vars:
            chunki = mdata.chunki_states
            fpath = odir / (base_name + f"_{var}" + f"_{chunki:06d}" + suffix)
            if algo.verbosity > 0:
                print("Writing file", fpath)

            o = FlowPlots2D(algo, farm_results)
            precalc = o.precalc_chunk_xy(
                var,
                mdata,
                fdata,
                resolution=resolution,
                n_img_points=n_img_points,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                z=z,
                xspace=xspace,
                yspace=yspace,
            )

            fig, ax = plt.subplots(figsize=figsize)
            anim = Animator(fig=fig)
            anim.add_generator(
                o.gen_states_fig_xy(
                    var=var,
                    fig=fig,
                    ax=ax,
                    animated=True,
                    ret_im=True,
                    precalc=precalc,
                    **kwargs,
                )
            )
            ani = anim.animate(verbosity=0)
            plt.close(fig)
            del precalc, fig, ax, anim

            if fpath.suffix == ".gif":
                ani.save(filename=fpath, writer="pillow", fps=fps)
            else:
                ani.save(filename=fpath, writer="ffmpeg", fps=fps)

    # case calc_points:
    elif mdata is not None and fdata is not None and tdata is not None:
        raise NotImplementedError(
            "Chunk animation writing not implemented for point calculations"
        )
    else:
        raise NotImplementedError(
            "Chunk animation writing not implemented for this case"
        )
