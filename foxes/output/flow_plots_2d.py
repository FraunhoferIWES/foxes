import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .output import Output
from foxes.utils import wd2uv
import foxes.constants as FC
import foxes.variables as FV


class FlowPlots2D(Output):
    """
    Class for horizontal or vertical 2D flow plots

    Attributes
    ----------
    algo: foxes.Algorithm
        The algorithm for point calculation
    farm_results: xarray.Dataset
        The farm results
    runner: foxes.utils.runners.Runner, optional
        The runner

    :group: output

    """

    def __init__(self, algo, farm_results, runner=None):
        """
        Constructor.

        Parameters
        ----------
        algo: foxes.Algorithm
            The algorithm for point calculation
        farm_results: xarray.Dataset
            The farm results
        runner: foxes.utils.runners.Runner, optional
            The runner

        """
        self.algo = algo
        self.fres = farm_results
        self.runner = runner

    def _get_fig(
        self,
        var,
        fig,
        figsize,
        ax,
        data,
        si,
        s,
        N_x,
        N_y,
        normalize_var,
        levels,
        x_pos,
        y_pos,
        vmin,
        vmax,
        cmap,
        xlabel,
        ylabel,
        title,
        add_bar,
        vlabel,
        ret_state,
        ret_im,
        quiv=None,
        invert_axis=None,
        animated=False,
    ):
        """
        Helper function for image creation
        """
        # create plot:
        if fig is None:
            hfig = plt.figure(figsize=figsize)
        else:
            hfig = fig
        if ax is None:
            hax = hfig.add_subplot(111)
        else:
            hax = ax

        # get results:
        zz = data[si].reshape([N_x, N_y]).T
        if normalize_var is not None:
            zz /= normalize_var

        # raw data image:
        if levels is None:
            im = hax.pcolormesh(
                x_pos,
                y_pos,
                zz,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
                cmap=cmap,
                animated=animated,
            )

        # contour plot:
        else:
            if vmax is not None and vmin is not None and not isinstance(levels, list):
                lvls = np.linspace(vmin, vmax, levels+1)
            else:
                lvls = levels
            im = hax.contourf(
                x_pos,
                y_pos,
                zz,
                levels=lvls,
                vmax=vmax,
                vmin=vmin,
                cmap=cmap,
                animated=animated,
            )

        qv = None
        if quiv is not None and quiv[0] is not None:
            n, pars, wd, ws = quiv
            uv = wd2uv(wd[si], ws[si])
            u = uv[:, 0].reshape([N_x, N_y]).T[::n, ::n]
            v = uv[:, 1].reshape([N_x, N_y]).T[::n, ::n]
            qv = hax.quiver(x_pos[::n], y_pos[::n], u, v, animated=animated, **pars)
            del n, pars, u, v, uv

        hax.autoscale_view()
        hax.set_xlabel(xlabel)
        hax.set_ylabel(ylabel)
        hax.set_aspect("equal", adjustable="box")

        ttl = None
        if animated and title is None:
            if hasattr(s, "dtype") and np.issubdtype(s.dtype, np.datetime64):
                t = np.datetime_as_string(s, unit="m").replace("T", " ")
            else:
                t = s
            ttl = hax.text(
                0.5,
                1.05,
                f"State {t}",
                backgroundcolor="w",
                transform=hax.transAxes,
                ha="center",
                animated=True,
                clip_on=False,
            )
        else:
            hax.set_title(title if title is not None else f"State {s}")

        if invert_axis == "x":
            hax.invert_xaxis()
        elif invert_axis == "y":
            hax.invert_yaxis()

        if add_bar:
            divider = make_axes_locatable(hax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            vlab = vlabel if vlabel is not None else var
            hfig.colorbar(im, cax=cax, orientation="vertical", label=vlab)
            out = hfig
        else:
            out = fig

        if ret_state or ret_im:
            out = [out]
        if ret_state:
            out.append(si)
        if ret_im:
            out.append([i for i in [im, qv, ttl] if i is not None])
        if ret_state or ret_im:
            out = tuple(out)

        return out

    def _calc_point_results(self, verbosity, g_pts, **kwargs):
        """Helper function for point data calculation"""
        averb = None if verbosity == self.algo.verbosity else self.algo.verbosity
        if averb is not None:
            self.algo.verbosity = verbosity
        if self.runner is None:
            point_results = self.algo.calc_points(self.fres, points=g_pts, **kwargs)
        else:
            kwargs["points"] = g_pts
            point_results = self.runner.run(
                self.algo.calc_points, args=(self.fres,), kwargs=kwargs
            )
        if averb is not None:
            self.algo.verbosity = averb

        return point_results

    def get_mean_fig_xy(
        self,
        var,
        resolution,
        xmin=None,
        ymin=None,
        xmax=None,
        ymax=None,
        xlabel="x [m]",
        ylabel="y [m]",
        z=None,
        xspace=500.0,
        yspace=500.0,
        levels=None,
        var_min=None,
        var_max=None,
        figsize=None,
        normalize_xy=None,
        normalize_var=None,
        title="States mean",
        vlabel=None,
        fig=None,
        ax=None,
        add_bar=True,
        cmap=None,
        weight_turbine=0,
        verbosity=0,
        ret_state=False,
        ret_im=False,
        animated=False,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a horizontal xy-plane.

        Parameters
        ----------
        var: str
            The variable name
        resolution: float
            The resolution in m
        xmin: float
            The min x coordinate, or None for automatic
        ymin: float
            The min y coordinate, or None for automatic
        xmax: float
            The max x coordinate, or None for automatic
        ymax: float
            The max y coordinate, or None for automatic
        xlabel: str
            The x axis label
        ylabel: str
            The y axis label
        z: float
            The z coordinate of the plane
        xspace: float
            The extra space in x direction, before and after wind farm
        yspace: float
            The extra space in y direction, before and after wind farm
        levels: int
            The number of levels for the contourf plot, or None for pure image
        var_min: float
            Minimum variable value
        var_max: float
            Maximum variable value
        figsize: tuple
            The figsize for plt.Figure
        normalize_xy: float, optional
            Divide x and y by this value
        normalize_var: float, optional
            Divide the variable by this value
        title: str, optional
            The title
        vlabel: str, optional
            The variable label
        fig: plt.Figure, optional
            The figure object
        ax: plt.Axes, optional
            The figure axes
        add_bar: bool
            Add a color bar
        cmap: str, optional
            The colormap
        weight_turbine: int, optional
            Index of the turbine from which to take the weight
        verbosity: int, optional
            The verbosity level
        ret_state: bool
            Flag for state index return
        ret_im: bool
            Flag for image return
        animated: bool
            Switch for usage for an animation
        kwargs: dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Yields
        ------
        fig: matplotlib.Figure
            The figure object
        si: int, optional
            The state index
        im: tuple, optional
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet

        """

        # prepare:
        n_states = self.algo.n_states

        # get base rectangle:
        x_min = xmin if xmin is not None else self.fres[FV.X].min().to_numpy() - xspace
        y_min = ymin if ymin is not None else self.fres[FV.Y].min().to_numpy() - yspace
        z_min = z if z is not None else self.fres[FV.H].min().to_numpy()
        x_max = xmax if xmax is not None else self.fres[FV.X].max().to_numpy() + xspace
        y_max = ymax if ymax is not None else self.fres[FV.Y].max().to_numpy() + yspace
        z_max = z if z is not None else self.fres[FV.H].max().to_numpy()

        x_pos, x_res = np.linspace(
            x_min,
            x_max,
            num=int((x_max - x_min) / resolution) + 1,
            endpoint=True,
            retstep=True,
            dtype=None,
        )
        y_pos, y_res = np.linspace(
            y_min,
            y_max,
            num=int((y_max - y_min) / resolution) + 1,
            endpoint=True,
            retstep=True,
            dtype=None,
        )
        N_x, N_y = len(x_pos), len(y_pos)
        n_pts = len(x_pos) * len(y_pos)
        z_pos = 0.5 * (z_min + z_max)
        g_pts = np.zeros((n_states, N_x, N_y, 3), dtype=FC.DTYPE)
        g_pts[:, :, :, 0] = x_pos[None, :, None]
        g_pts[:, :, :, 1] = y_pos[None, None, :]
        g_pts[:, :, :, 2] = z_pos
        g_pts = g_pts.reshape(n_states, n_pts, 3)

        if verbosity > 0:
            print("\nFlowPlots2D plot grid:")
            print("Min XYZ  =", x_min, y_min, z_min)
            print("Max XYZ  =", x_max, y_max, z_max)
            print("Pos Z    =", z_pos)
            print("Res XY   =", x_res, y_res)
            print("Dim XY   =", N_x, N_y)
            print("Grid pts =", n_pts)

        # calculate point results:
        data = self._calc_point_results(verbosity, g_pts, **kwargs)[var].to_numpy()

        # take mean over states:
        weights = self.fres[FV.WEIGHT][:, weight_turbine].to_numpy()
        data = np.einsum("s,sp->p", weights, data)

        # find data min max:
        vmin = var_min if var_min is not None else np.nanmin(data)
        vmax = var_max if var_max is not None else np.nanmax(data)
        if normalize_var is not None:
            vmin /= normalize_var
            vmax /= normalize_var

        # normalize x and y:
        if normalize_xy is not None:
            x_pos /= normalize_xy
            y_pos /= normalize_xy

        if title is None:
            title = f"States mean, z =  {int(np.round(z_pos))} m"

        # create plot:
        out = self._get_fig(
            var,
            fig,
            figsize,
            ax,
            data,
            None,
            None,
            N_x,
            N_y,
            normalize_var,
            levels,
            x_pos,
            y_pos,
            vmin,
            vmax,
            cmap,
            xlabel,
            ylabel,
            title,
            add_bar,
            vlabel,
            ret_state,
            ret_im,
            animated=animated,
        )

        return out

    def get_mean_fig_xz(
        self,
        var,
        resolution,
        x_direction=270,
        xmin=None,
        zmin=0.0,
        xmax=None,
        zmax=None,
        xlabel="x [m]",
        zlabel="z [m]",
        y=None,
        xspace=500.0,
        zspace=500.0,
        levels=None,
        var_min=None,
        var_max=None,
        figsize=None,
        normalize_x=None,
        normalize_z=None,
        normalize_var=None,
        title=None,
        vlabel=None,
        fig=None,
        ax=None,
        add_bar=True,
        cmap=None,
        weight_turbine=0,
        verbosity=0,
        ret_state=False,
        ret_im=False,
        animated=False,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a vertical xz-plane.

        Parameters
        ----------
        var: str
            The variable name
        resolution: float
            The resolution in m
        x_direction: float
            The direction of the x axis, 0 = north
        xmin: float
            The min x coordinate, or None for automatic
        zmin: float
            The min z coordinate
        xmax: float
            The max x coordinate, or None for automatic
        zmax: float
            The max z coordinate, or None for automatic
        xlabel: str
            The x axis label
        zlabel: str
            The z axis label
        y: float
            The y coordinate of the plane
        xspace: float
            The extra space in x direction, before and after wind farm
        zspace: float
            The extra space in z direction, below and above wind farm
        levels: int
            The number of levels for the contourf plot, or None for pure image
        var_min: float
            Minimum variable value
        var_max: float
            Maximum variable value
        figsize: tuple
            The figsize for plt.Figure
        normalize_x: float, optional
            Divide x by this value
        normalize_z: float, optional
            Divide z by this value
        normalize_var: float, optional
            Divide the variable by this value
        title: str, optional
            The title
        vlabel: str, optional
            The variable label
        fig: plt.Figure, optional
            The figure object
        ax: plt.Axes, optional
            The figure axes
        add_bar: bool
            Add a color bar
        cmap: str, optional
            The colormap
        weight_turbine: int, optional
            Index of the turbine from which to take the weight
        verbosity: int, optional
            The verbosity level
        ret_state: bool
            Flag for state index return
        ret_im: bool
            Flag for image return
        animated: bool
            Switch for usage for an animation
        kwargs: dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Yields
        ------
        fig: matplotlib.Figure
            The figure object
        si: int, optional
            The state index
        im: tuple, optional
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet

        """

        # prepare:
        n_states = self.algo.n_states
        n_turbines = self.algo.n_turbines
        n_x = np.append(wd2uv(x_direction), [0.0], axis=0)
        n_z = np.array([0.0, 0.0, 1.0])
        n_y = np.cross(n_z, n_x)

        # project to axes:
        xyz = np.zeros((n_states, n_turbines, 3), dtype=FC.DTYPE)
        xyz[:, :, 0] = self.fres[FV.X]
        xyz[:, :, 1] = self.fres[FV.Y]
        xyz[:, :, 2] = self.fres[FV.H]
        xx = np.einsum("std,d->st", xyz, n_x)
        yy = np.einsum("std,d->st", xyz, n_y)
        zz = np.einsum("std,d->st", xyz, n_z)
        del xyz

        # get base rectangle:
        x_min = xmin if xmin is not None else np.min(xx) - xspace
        z_min = zmin if zmin is not None else np.minimum(np.min(zz) - zspace, 0.0)
        y_min = y if y is not None else np.min(yy)
        x_max = xmax if xmax is not None else np.max(xx) + xspace
        z_max = zmax if zmax is not None else np.max(zz) + zspace
        y_max = y if y is not None else np.max(yy)
        del xx, yy, zz

        x_pos, x_res = np.linspace(
            x_min,
            x_max,
            num=int((x_max - x_min) / resolution) + 1,
            endpoint=True,
            retstep=True,
            dtype=None,
        )
        z_pos, z_res = np.linspace(
            z_min,
            z_max,
            num=int((z_max - z_min) / resolution) + 1,
            endpoint=True,
            retstep=True,
            dtype=None,
        )
        N_x, N_z = len(x_pos), len(z_pos)
        n_pts = len(x_pos) * len(z_pos)
        y_pos = 0.5 * (y_min + y_max)
        g_pts = np.zeros((n_states, N_x, N_z, 3), dtype=FC.DTYPE)
        g_pts[:] += x_pos[None, :, None, None] * n_x[None, None, None, :]
        g_pts[:] += y_pos * n_y[None, None, None, :]
        g_pts[:] += z_pos[None, None, :, None] * n_z[None, None, None, :]
        g_pts = g_pts.reshape(n_states, n_pts, 3)

        if verbosity > 0:
            print("\nFlowPlots2D plot grid:")
            print("Min XYZ  =", x_min, y_min, z_min)
            print("Max XYZ  =", x_max, y_max, z_max)
            print("Pos Y    =", y_pos)
            print("Res XZ   =", x_res, z_res)
            print("Dim XZ   =", N_x, N_z)
            print("Grid pts =", n_pts)

        # calculate point results:
        data = self._calc_point_results(verbosity, g_pts, **kwargs)[var].to_numpy()

        # take mean over states:
        weights = self.fres[FV.WEIGHT][:, weight_turbine].to_numpy()
        data = np.einsum("s,sp->p", weights, data)

        # find data min max:
        vmin = var_min if var_min is not None else np.nanmin(data)
        vmax = var_max if var_max is not None else np.nanmax(data)
        if normalize_var is not None:
            vmin /= normalize_var
            vmax /= normalize_var

        # normalize x and z:
        if normalize_x is not None:
            x_pos /= normalize_x
        if normalize_z is not None:
            z_pos /= normalize_z

        if title is None:
            title = f"States mean, x direction {x_direction}°, y =  {int(np.round(y_pos))} m"

        # create plot:
        out = self._get_fig(
            var,
            fig,
            figsize,
            ax,
            data,
            None,
            None,
            N_x,
            N_z,
            normalize_var,
            levels,
            x_pos,
            z_pos,
            vmin,
            vmax,
            cmap,
            xlabel,
            zlabel,
            title,
            add_bar,
            vlabel,
            ret_state,
            ret_im,
            animated=animated,
        )

        return out

    def get_mean_fig_yz(
        self,
        var,
        resolution,
        x_direction=270,
        ymin=None,
        zmin=0.0,
        ymax=None,
        zmax=None,
        ylabel="x [m]",
        zlabel="z [m]",
        x=None,
        yspace=500.0,
        zspace=500.0,
        levels=None,
        var_min=None,
        var_max=None,
        figsize=None,
        normalize_y=None,
        normalize_z=None,
        normalize_var=None,
        title=None,
        vlabel=None,
        fig=None,
        ax=None,
        add_bar=True,
        cmap=None,
        weight_turbine=0,
        verbosity=1,
        ret_state=False,
        ret_im=False,
        animated=False,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a vertical yz-plane.

        Parameters
        ----------
        var: str
            The variable name
        resolution: float
            The resolution in m
        x_direction: float
            The direction of the x axis, 0 = north
        ymin: float
            The min y coordinate, or None for automatic
        zmin: float
            The min z coordinate
        ymax: float
            The max y coordinate, or None for automatic
        zmax: float
            The max z coordinate, or None for automatic
        ylabel: str
            The y axis label
        zlabel: str
            The z axis label
        x: float
            The x coordinate of the plane
        yspace: float
            The extra space in y direction, before and after wind farm
        zspace: float
            The extra space in z direction, below and above wind farm
        levels: int
            The number of levels for the contourf plot, or None for pure image
        var_min: float
            Minimum variable value
        var_max: float
            Maximum variable value
        figsize: tuple
            The figsize for plt.Figure
        normalize_y: float, optional
            Divide y by this value
        normalize_z: float, optional
            Divide z by this value
        normalize_var: float, optional
            Divide the variable by this value
        title: str, optional
            The title
        vlabel: str, optional
            The variable label
        fig: plt.Figure, optional
            The figure object
        ax: plt.Axes, optional
            The figure axes
        add_bar: bool
            Add a color bar
        cmap: str, optional
            The colormap
        weight_turbine: int, optional
            Index of the turbine from which to take the weight
        verbosity: int, optional
            The verbosity level
        ret_state: bool
            Flag for state index return
        ret_im: bool
            Flag for image return
        animated: bool
            Switch for usage for an animation
        kwargs: dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Yields
        ------
        fig: matplotlib.Figure
            The figure object
        si: int, optional
            The state index
        im: tuple, optional
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet

        """

        # prepare:
        n_states = self.algo.n_states
        n_turbines = self.algo.n_turbines
        n_x = np.append(wd2uv(x_direction), [0.0], axis=0)
        n_z = np.array([0.0, 0.0, 1.0])
        n_y = np.cross(n_z, n_x)

        # project to axes:
        xyz = np.zeros((n_states, n_turbines, 3), dtype=FC.DTYPE)
        xyz[:, :, 0] = self.fres[FV.X]
        xyz[:, :, 1] = self.fres[FV.Y]
        xyz[:, :, 2] = self.fres[FV.H]
        xx = np.einsum("std,d->st", xyz, n_x)
        yy = np.einsum("std,d->st", xyz, n_y)
        zz = np.einsum("std,d->st", xyz, n_z)
        del xyz

        # get base rectangle:
        y_min = ymin if ymin is not None else np.min(yy) - yspace
        z_min = zmin if zmin is not None else np.minimum(np.min(zz) - zspace, 0.0)
        x_min = x if x is not None else np.min(xx)
        y_max = ymax if ymax is not None else np.max(yy) + yspace
        z_max = zmax if zmax is not None else np.max(zz) + zspace
        x_max = x if x is not None else np.max(xx)
        del xx, yy, zz

        y_pos, y_res = np.linspace(
            y_min,
            y_max,
            num=int((y_max - y_min) / resolution) + 1,
            endpoint=True,
            retstep=True,
            dtype=None,
        )
        z_pos, z_res = np.linspace(
            z_min,
            z_max,
            num=int((z_max - z_min) / resolution) + 1,
            endpoint=True,
            retstep=True,
            dtype=None,
        )
        N_y, N_z = len(y_pos), len(z_pos)
        n_pts = len(y_pos) * len(z_pos)
        x_pos = 0.5 * (x_min + x_max)
        g_pts = np.zeros((n_states, N_y, N_z, 3), dtype=FC.DTYPE)
        g_pts[:] += x_pos * n_x[None, None, None, :]
        g_pts[:] += y_pos[None, :, None, None] * n_y[None, None, None, :]
        g_pts[:] += z_pos[None, None, :, None] * n_z[None, None, None, :]
        g_pts = g_pts.reshape(n_states, n_pts, 3)

        if verbosity > 0:
            print("\nFlowPlots2D plot grid:")
            print("Min XYZ  =", x_min, y_min, z_min)
            print("Max XYZ  =", x_max, y_max, z_max)
            print("Pos X    =", x_pos)
            print("Res YZ   =", y_res, z_res)
            print("Dim YZ   =", N_y, N_z)
            print("Grid pts =", n_pts)

        # calculate point results:
        data = self._calc_point_results(verbosity, g_pts, **kwargs)[var].to_numpy()

        # take mean over states:
        weights = self.fres[FV.WEIGHT][:, weight_turbine].to_numpy()
        data = np.einsum("s,sp->p", weights, data)

        # find data min max:
        vmin = var_min if var_min is not None else np.nanmin(data)
        vmax = var_max if var_max is not None else np.nanmax(data)
        if normalize_var is not None:
            vmin /= normalize_var
            vmax /= normalize_var

        # normalize x and z:
        if normalize_y is not None:
            y_pos /= normalize_y
        if normalize_z is not None:
            z_pos /= normalize_z

        if title is None:
            title = f"States mean, x direction {x_direction}°, x =  {int(np.round(x_pos))} m"

        # create plot:
        out = self._get_fig(
            var,
            fig,
            figsize,
            ax,
            data,
            None,
            None,
            N_y,
            N_z,
            normalize_var,
            levels,
            y_pos,
            z_pos,
            vmin,
            vmax,
            cmap,
            ylabel,
            zlabel,
            title,
            add_bar,
            vlabel,
            ret_state,
            ret_im,
            invert_axis="x",
            animated=animated,
        )

        return out

    def gen_states_fig_xy(
        self,
        var,
        resolution,
        xmin=None,
        ymin=None,
        xmax=None,
        ymax=None,
        xlabel="x [m]",
        ylabel="y [m]",
        z=None,
        xspace=500.0,
        yspace=500.0,
        levels=None,
        var_min=None,
        var_max=None,
        figsize=None,
        normalize_xy=None,
        normalize_var=None,
        title=None,
        vlabel=None,
        fig=None,
        ax=None,
        add_bar=True,
        cmap=None,
        quiver_n=None,
        quiver_pars={},
        verbosity=0,
        ret_state=False,
        ret_im=False,
        animated=False,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a horizontal xy-plane.

        Parameters
        ----------
        var: str
            The variable name
        resolution: float
            The resolution in m
        xmin: float
            The min x coordinate, or None for automatic
        ymin: float
            The min y coordinate, or None for automatic
        xmax: float
            The max x coordinate, or None for automatic
        ymax: float
            The max y coordinate, or None for automatic
        xlabel: str
            The x axis label
        ylabel: str
            The y axis label
        z: float
            The z coordinate of the plane
        xspace: float
            The extra space in x direction, before and after wind farm
        yspace: float
            The extra space in y direction, before and after wind farm
        levels: int
            The number of levels for the contourf plot, or None for pure image
        var_min: float
            Minimum variable value
        var_max: float
            Maximum variable value
        figsize: tuple
            The figsize for plt.Figure
        normalize_xy: float, optional
            Divide x and y by this value
        normalize_var: float, optional
            Divide the variable by this value
        title: str, optional
            The title
        vlabel: str, optional
            The variable label
        fig: plt.Figure, optional
            The figure object
        ax: plt.Axes, optional
            The figure axes
        add_bar: bool
            Add a color bar
        cmap: str, optional
            The colormap
        quiver_n: int, optional
            Place a vector at each `n`th point
        quiver_pars: dict, optional
            Parameters for plt.quiver
        verbosity: int, optional
            The verbosity level
        ret_state: bool
            Flag for state index return
        ret_im: bool
            Flag for image return
        animated: bool
            Switch for usage for an animation
        kwargs: dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Yields
        ------
        fig: matplotlib.Figure
            The figure object
        si: int, optional
            The state index
        im: tuple, optional
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet

        """

        # prepare:
        n_states = self.algo.n_states

        # get base rectangle:
        x_min = xmin if xmin is not None else self.fres[FV.X].min().to_numpy() - xspace
        y_min = ymin if ymin is not None else self.fres[FV.Y].min().to_numpy() - yspace
        z_min = z if z is not None else self.fres[FV.H].min().to_numpy()
        x_max = xmax if xmax is not None else self.fres[FV.X].max().to_numpy() + xspace
        y_max = ymax if ymax is not None else self.fres[FV.Y].max().to_numpy() + yspace
        z_max = z if z is not None else self.fres[FV.H].max().to_numpy()

        x_pos, x_res = np.linspace(
            x_min,
            x_max,
            num=int((x_max - x_min) / resolution) + 1,
            endpoint=True,
            retstep=True,
            dtype=None,
        )
        y_pos, y_res = np.linspace(
            y_min,
            y_max,
            num=int((y_max - y_min) / resolution) + 1,
            endpoint=True,
            retstep=True,
            dtype=None,
        )
        N_x, N_y = len(x_pos), len(y_pos)
        n_pts = len(x_pos) * len(y_pos)
        z_pos = 0.5 * (z_min + z_max)
        g_pts = np.zeros((n_states, N_x, N_y, 3), dtype=FC.DTYPE)
        g_pts[:, :, :, 0] = x_pos[None, :, None]
        g_pts[:, :, :, 1] = y_pos[None, None, :]
        g_pts[:, :, :, 2] = z_pos
        g_pts = g_pts.reshape(n_states, n_pts, 3)

        if verbosity > 0:
            print("\nFlowPlots2D plot grid:")
            print("Min XYZ  =", x_min, y_min, z_min)
            print("Max XYZ  =", x_max, y_max, z_max)
            print("Pos Z    =", z_pos)
            print("Res XY   =", x_res, y_res)
            print("Dim XY   =", N_x, N_y)
            print("Grid pts =", n_pts)

        # calculate point results:
        point_results = self._calc_point_results(verbosity, g_pts, **kwargs)
        data = point_results[var].to_numpy()
        quiv = (
            None
            if quiver_n is None
            else (
                quiver_n,
                quiver_pars,
                point_results[FV.WD].values,
                point_results[FV.WS].values,
            )
        )
        del point_results

        # find data min max:
        vmin = var_min if var_min is not None else np.nanmin(data)
        vmax = var_max if var_max is not None else np.nanmax(data)
        if normalize_var is not None:
            vmin /= normalize_var
            vmax /= normalize_var

        # normalize x and y:
        if normalize_xy is not None:
            x_pos /= normalize_xy
            y_pos /= normalize_xy

        # loop over states:
        for si, s in enumerate(self.fres[FC.STATE].to_numpy()):
            if not animated and title is None:
                ttl = f"State {s}"
                ttl += f", z =  {int(np.round(z_pos))} m"
            else:
                ttl = title

            out = self._get_fig(
                var,
                fig,
                figsize,
                ax,
                data,
                si,
                s,
                N_x,
                N_y,
                normalize_var,
                levels,
                x_pos,
                y_pos,
                vmin,
                vmax,
                cmap,
                xlabel,
                ylabel,
                ttl,
                add_bar,
                vlabel,
                ret_state,
                ret_im,
                quiv,
                animated=animated,
            )

            yield out

    def gen_states_fig_xz(
        self,
        var,
        resolution,
        x_direction=270.0,
        xmin=None,
        zmin=0.0,
        xmax=None,
        zmax=None,
        xlabel="x [m]",
        zlabel="z [m]",
        y=None,
        xspace=500.0,
        zspace=500.0,
        levels=None,
        var_min=None,
        var_max=None,
        figsize=None,
        normalize_x=None,
        normalize_z=None,
        normalize_var=None,
        title=None,
        vlabel=None,
        fig=None,
        ax=None,
        add_bar=True,
        cmap=None,
        quiver_n=None,
        quiver_pars={},
        verbosity=0,
        ret_state=False,
        ret_im=False,
        animated=False,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a vertical xz-plane.

        Parameters
        ----------
        var: str
            The variable name
        resolution: float
            The resolution in m
        x_direction: float
            The direction of the x axis, 0 = north
        xmin: float
            The min x coordinate, or None for automatic
        zmin: float
            The min z coordinate
        xmax: float
            The max x coordinate, or None for automatic
        zmax: float
            The max z coordinate, or None for automatic
        xlabel: str
            The x axis label
        zlabel: str
            The z axis label
        y: float
            The y coordinate of the plane
        xspace: float
            The extra space in x direction, before and after wind farm
        zspace: float
            The extra space in z direction, below and above wind farm
        levels: int
            The number of levels for the contourf plot, or None for pure image
        var_min: float
            Minimum variable value
        var_max: float
            Maximum variable value
        figsize: tuple
            The figsize for plt.Figure
        normalize_x: float, optional
            Divide x by this value
        normalize_z: float, optional
            Divide z by this value
        normalize_var: float, optional
            Divide the variable by this value
        title: str, optional
            The title
        vlabel: str, optional
            The variable label
        fig: plt.Figure, optional
            The figure object
        ax: plt.Axes, optional
            The figure axes
        add_bar: bool
            Add a color bar
        cmap: str, optional
            The colormap
        quiver_n: int, optional
            Place a vector at ech `n`th point
        quiver_pars: dict, optional
            Parameters for plt.quiver
        verbosity: int, optional
            The verbosity level
        ret_state: bool
            Flag for state index return
        ret_im: bool
            Flag for image return
        animated: bool
            Switch for usage for an animation
        kwargs: dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Yields
        ------
        fig: matplotlib.Figure
            The figure object
        si: int, optional
            The state index
        im: tuple, optional
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet

        """

        # prepare:
        n_states = self.algo.n_states
        n_turbines = self.algo.n_turbines
        n_x = np.append(wd2uv(x_direction), [0.0], axis=0)
        n_z = np.array([0.0, 0.0, 1.0])
        n_y = np.cross(n_z, n_x)

        # project to axes:
        xyz = np.zeros((n_states, n_turbines, 3), dtype=FC.DTYPE)
        xyz[:, :, 0] = self.fres[FV.X]
        xyz[:, :, 1] = self.fres[FV.Y]
        xyz[:, :, 2] = self.fres[FV.H]
        xx = np.einsum("std,d->st", xyz, n_x)
        yy = np.einsum("std,d->st", xyz, n_y)
        zz = np.einsum("std,d->st", xyz, n_z)
        del xyz

        # get base rectangle:
        x_min = xmin if xmin is not None else np.min(xx) - xspace
        z_min = zmin if zmin is not None else np.minimum(np.min(zz) - zspace, 0.0)
        y_min = y if y is not None else np.min(yy)
        x_max = xmax if xmax is not None else np.max(xx) + xspace
        z_max = zmax if zmax is not None else np.max(zz) + zspace
        y_max = y if y is not None else np.max(yy)
        del xx, yy, zz

        x_pos, x_res = np.linspace(
            x_min,
            x_max,
            num=int((x_max - x_min) / resolution) + 1,
            endpoint=True,
            retstep=True,
            dtype=None,
        )
        z_pos, z_res = np.linspace(
            z_min,
            z_max,
            num=int((z_max - z_min) / resolution) + 1,
            endpoint=True,
            retstep=True,
            dtype=None,
        )
        N_x, N_z = len(x_pos), len(z_pos)
        n_pts = len(x_pos) * len(z_pos)
        y_pos = 0.5 * (y_min + y_max)
        g_pts = np.zeros((n_states, N_x, N_z, 3), dtype=FC.DTYPE)
        g_pts[:] += x_pos[None, :, None, None] * n_x[None, None, None, :]
        g_pts[:] += y_pos * n_y[None, None, None, :]
        g_pts[:] += z_pos[None, None, :, None] * n_z[None, None, None, :]
        g_pts = g_pts.reshape(n_states, n_pts, 3)

        if verbosity > 0:
            print("\nFlowPlots2D plot grid:")
            print("Min XYZ  =", x_min, y_min, z_min)
            print("Max XYZ  =", x_max, y_max, z_max)
            print("Pos Y    =", y_pos)
            print("Res XZ   =", x_res, z_res)
            print("Dim XZ   =", N_x, N_z)
            print("Grid pts =", n_pts)

        # calculate point results:
        point_results = self._calc_point_results(verbosity, g_pts, **kwargs)
        data = point_results[var].to_numpy()
        quiv = (
            None
            if quiver_n is None
            else (
                quiver_n,
                quiver_pars,
                point_results[FV.WD].values,
                point_results[FV.WS].values,
            )
        )
        del point_results

        # find data min max:
        vmin = var_min if var_min is not None else np.nanmin(data)
        vmax = var_max if var_max is not None else np.nanmax(data)
        if normalize_var is not None:
            vmin /= normalize_var
            vmax /= normalize_var

        # normalize x and z:
        if normalize_x is not None:
            x_pos /= normalize_x
        if normalize_z is not None:
            z_pos /= normalize_z

        # loop over states:
        for si, s in enumerate(self.fres[FC.STATE].to_numpy()):
            if not animated and title is None:
                ttl = f"State {s}"
                ttl += f", x direction = {x_direction}°"
                ttl += f", y =  {int(np.round(y_pos))} m"
            else:
                ttl = title

            out = self._get_fig(
                var,
                fig,
                figsize,
                ax,
                data,
                si,
                s,
                N_x,
                N_z,
                normalize_var,
                levels,
                x_pos,
                z_pos,
                vmin,
                vmax,
                cmap,
                xlabel,
                zlabel,
                ttl,
                add_bar,
                vlabel,
                ret_state,
                ret_im,
                quiv,
                animated=animated,
            )

            yield out

    def gen_states_fig_yz(
        self,
        var,
        resolution,
        x_direction=270.0,
        ymin=None,
        zmin=0.0,
        ymax=None,
        zmax=None,
        ylabel="y [m]",
        zlabel="z [m]",
        x=None,
        yspace=500.0,
        zspace=500.0,
        levels=None,
        var_min=None,
        var_max=None,
        figsize=None,
        normalize_y=None,
        normalize_z=None,
        normalize_var=None,
        title=None,
        vlabel=None,
        fig=None,
        ax=None,
        add_bar=True,
        cmap=None,
        quiver_n=None,
        quiver_pars={},
        verbosity=1,
        ret_state=False,
        ret_im=False,
        animated=False,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a plane.

        Parameters
        ----------
        var: str
            The variable name
        resolution: float
            The resolution in m
        x_direction: float
            The direction of the x axis, 0 = north
        ymin: float
            The min y coordinate, or None for automatic
        zmin: float
            The min z coordinate
        ymax: float
            The max y coordinate, or None for automatic
        zmax: float
            The max z coordinate, or None for automatic
        ylabel: str
            The y axis label
        zlabel: str
            The z axis label
        x: float
            The x coordinate of the plane
        yspace: float
            The extra space in y direction, left and right of wind farm
        zspace: float
            The extra space in z direction, below and above wind farm
        levels: int
            The number of levels for the contourf plot, or None for pure image
        var_min: float
            Minimum variable value
        var_max: float
            Maximum variable value
        figsize: tuple
            The figsize for plt.Figure
        normalize_y: float, optional
            Divide y by this value
        normalize_z: float, optional
            Divide z by this value
        normalize_var: float, optional
            Divide the variable by this value
        title: str, optional
            The title
        vlabel: str, optional
            The variable label
        fig: plt.Figure, optional
            The figure object
        ax: plt.Axes, optional
            The figure axes
        add_bar: bool
            Add a color bar
        cmap: str, optional
            The colormap
        quiver_n: int, optional
            Place a vector at ech `n`th point
        quiver_pars: dict, optional
            Parameters for plt.quiver
        verbosity: int, optional
            The verbosity level
        ret_state: bool
            Flag for state index return
        ret_im: bool
            Flag for image return
        animated: bool
            Switch for usage for an animation
        kwargs: dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Yields
        ------
        fig: matplotlib.Figure
            The figure object
        si: int, optional
            The state index
        im: tuple, optional
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet

        """

        # prepare:
        n_states = self.algo.n_states
        n_turbines = self.algo.n_turbines
        n_x = np.append(wd2uv(x_direction), [0.0], axis=0)  ## -180 to get [1,0,0]
        n_z = np.array([0.0, 0.0, 1.0])
        n_y = np.cross(n_z, n_x)

        # project to axes:
        xyz = np.zeros((n_states, n_turbines, 3), dtype=FC.DTYPE)
        xyz[:, :, 0] = self.fres[FV.X]
        xyz[:, :, 1] = self.fres[FV.Y]
        xyz[:, :, 2] = self.fres[FV.H]
        xx = np.einsum("std,d->st", xyz, n_x)
        yy = np.einsum("std,d->st", xyz, n_y)
        zz = np.einsum("std,d->st", xyz, n_z)
        del xyz

        # get base rectangle:
        y_min = ymin if ymin is not None else np.min(yy) - yspace
        z_min = zmin if zmin is not None else np.minimum(np.min(zz) - zspace, 10.0)
        x_min = x if x is not None else np.min(xx)
        y_max = ymax if ymax is not None else np.max(yy) + yspace
        z_max = zmax if zmax is not None else np.max(zz) + zspace
        x_max = x if x is not None else np.max(xx)
        del xx, yy, zz

        y_pos, y_res = np.linspace(
            y_min,
            y_max,
            num=int((y_max - y_min) / resolution) + 1,
            endpoint=True,
            retstep=True,
            dtype=None,
        )
        z_pos, z_res = np.linspace(
            z_min,
            z_max,
            num=int((z_max - z_min) / resolution) + 1,
            endpoint=True,
            retstep=True,
            dtype=None,
        )
        N_y, N_z = len(y_pos), len(z_pos)
        n_pts = len(y_pos) * len(z_pos)
        x_pos = 0.5 * (x_min + x_max)
        g_pts = np.zeros((n_states, N_y, N_z, 3), dtype=FC.DTYPE)
        g_pts[:] += x_pos * n_x[None, None, None, :]
        g_pts[:] += y_pos[None, :, None, None] * n_y[None, None, None, :]
        g_pts[:] += z_pos[None, None, :, None] * n_z[None, None, None, :]
        g_pts = g_pts.reshape(n_states, n_pts, 3)

        if verbosity > 0:
            print("\nFlowPlots2D plot grid:")
            print("Min XYZ  =", x_min, y_min, z_min)
            print("Max XYZ  =", x_max, y_max, z_max)
            print("Pos X    =", x_pos)
            print("Res YZ   =", y_res, z_res)
            print("Dim YZ   =", N_y, N_z)
            print("Grid pts =", n_pts)

        # calculate point results:
        point_results = self._calc_point_results(verbosity, g_pts, **kwargs)
        data = point_results[var].to_numpy()
        quiv = (
            None
            if quiver_n is None
            else (
                quiver_n,
                quiver_pars,
                point_results[FV.WD].values,
                point_results[FV.WS].values,
            )
        )
        del point_results

        # find data min max:
        vmin = var_min if var_min is not None else np.nanmin(data)
        vmax = var_max if var_max is not None else np.nanmax(data)
        if normalize_var is not None:
            vmin /= normalize_var
            vmax /= normalize_var

        # normalize x and z:
        if normalize_y is not None:
            y_pos /= normalize_y
        if normalize_z is not None:
            z_pos /= normalize_z

        # loop over states:
        for si, s in enumerate(self.fres[FC.STATE].to_numpy()):
            ttl = f"State {s}" if title is None else title
            ttl += f", x direction = {x_direction}°"
            ttl += f", x =  {int(np.round(x_pos))} m"

            out = self._get_fig(
                var,
                fig,
                figsize,
                ax,
                data,
                si,
                s,
                N_y,
                N_z,
                normalize_var,
                levels,
                y_pos,
                z_pos,
                vmin,
                vmax,
                cmap,
                ylabel,
                zlabel,
                ttl,
                add_bar,
                vlabel,
                ret_state,
                ret_im,
                quiv=quiv,
                invert_axis="x",
                animated=animated,
            )

            yield out
