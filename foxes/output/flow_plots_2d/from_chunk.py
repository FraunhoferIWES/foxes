import numpy as np
from xarray import Dataset
from pathlib import Path
import matplotlib.pyplot as plt

from foxes.config import get_output_path
from foxes.core import TData
import foxes.constants as FC
import foxes.variables as FV

from .flow_plots import FlowPlots2D
from ..animation import Animator
from ..grids import get_grid_xy, np2np_sp

def write_chunk_ani_xy(
    algo, 
    mdata, 
    fdata, 
    tdata=None,
    vars=[FV.WS],
    resolution=100,
    figsize=(8, 8),
    fpath_base="chunk_animation.gif",
    n_img_points=None,
    xmin=None,
    ymin=None,
    xmax=None,
    ymax=None,
    z=None,
    xspace=500.0,
    yspace=500.0,
    states_sel=None,
    states_isel=None,
    fps=4,
    **kwargs,
):
    """
    Writes an animation of a chunk calculation to file.
    
    Parameters
    ----------
    algo: foxes.core.Algorithm
        The calculation algorithm
    mdata: foxes.core.MData
        The model data
    fdata: foxes.core.FData
        The farm data
    tdata: foxes.core.TData, optional
        The point data, for point calculations
    vars: list of str
        The variables to be plotted
    resolution: float
        The resolution of the plot
    figsize: tuple of float
        The figure size
    fpath_base: str
        The base name for the output files, including suffix,
        e.g. 'output/chunk_ani.gif' or 'output/chunk_ani.mp4'
    n_img_points: int, optional
        The number of image points, or `None` for automatic
    xmin: float, optional
        The minimum x coordinate, or `None` for automatic
    ymin: float, optional
        The minimum y coordinate, or `None` for automatic
    xmax: float, optional
        The maximum x coordinate, or `None` for automatic
    ymax: float, optional
        The maximum y coordinate, or `None` for automatic
    z: float, optional
        The z coordinate of the slice, or `None` for automatic
    xspace: float
        The spacing in x direction if xmin/xmax are automatic
    yspace: float
        The spacing in y direction if ymin/ymax are automatic
    states_sel: list, optional
        Reduce to selected states
    states_isel: list, optional
        Reduce to the selected states indices
    fps: int
        The frames per second for the animation
    kwargs: dict
        Additional keyword arguments for the plotting function

    """
    # case calc_farm:
    if mdata is not None and fdata is not None and tdata is None:

        try:
            if states_isel is not None:
                mdata = mdata.get_slice(FC.STATE, states_isel, force=True)
                fdata = fdata.get_slice(FC.STATE, states_isel, force=True)
            if states_sel is not None:
                s = [i for i in range(mdata.n_states) if mdata[FC.STATE][i] in states_sel]
                mdata = mdata.get_slice(FC.STATE, s, force=True)
                fdata = fdata.get_slice(FC.STATE, s, force=True)
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
        sinds = mdata[FC.STATE]
        n_states = len(sinds)

        fpath_base = get_output_path(fpath_base)
        odir = fpath_base.parent
        odir.mkdir(parents=True, exist_ok=True)
        base_name = fpath_base.stem
        suffix = fpath_base.suffix

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
            anim.add_generator(o.gen_states_fig_xy(
                var=var, 
                fig=fig, 
                ax=ax, 
                animated=True, 
                ret_im=True,
                precalc=precalc,
                **kwargs,
            ))
            ani = anim.animate(verbosity=0)
            plt.close(fig)
            del precalc, fig, ax, anim

            if fpath.suffix == ".gif":
                ani.save(filename=fpath, writer="pillow", fps=fps)
            else:
                ani.save(filename=fpath, writer="ffmpeg", fps=fps)

    # case calc_points:
    elif mdata is not None and fdata is not None and tdata is not None:
        raise NotImplementedError("Chunk animation writing not implemented for point calculations")
    else:
        raise NotImplementedError("Chunk animation writing not implemented for this case")
