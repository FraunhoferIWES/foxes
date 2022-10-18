import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .output import Output
from foxes.utils import wd2wdvec, wd2uv
import foxes.constants as FC
import foxes.variables as FV


class FlowPlots2D(Output):
    """
    Class for horizontal or vertical 2D flow plots

    Parameters
    ----------
    algo : foxes.Algorithm
        The algorithm for point calculation
    farm_results : xarray.Dataset
        The farm results

    Attributes
    ----------
    algo : foxes.Algorithm
        The algorithm for point calculation
    farm_results : xarray.Dataset
        The farm results

    """

    def __init__(self, algo, farm_results):
        self.algo = algo
        self.fres = farm_results

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
                x_pos, y_pos, zz, vmin=vmin, vmax=vmax, shading="auto", cmap=cmap
            )

        # contour plot:
        else:
            im = hax.contourf(x_pos, y_pos, zz, levels, vmax=vmax, vmin=vmin, cmap=cmap)

        if quiv is not None and quiv[0] is not None:
            n, pars, wd, ws = quiv
            uv = wd2uv(wd[si], ws[si])
            u = uv[:, 0].reshape([N_x, N_y]).T[::n, ::n]
            v = uv[:, 1].reshape([N_x, N_y]).T[::n, ::n]
            hax.quiver(x_pos[::n], y_pos[::n], u, v, **pars)
            del n, pars, u, v, uv

        hax.autoscale_view()
        hax.set_xlabel(xlabel)
        hax.set_ylabel(ylabel)
        hax.set_title(title if title is not None else f"State {s}")
        hax.set_aspect("equal", adjustable="box")

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
            out.append(im)
        if ret_state or ret_im:
            out = tuple(out)

        return out

    def get_mean_fig_horizontal(
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
        verbosity=1,
        ret_state=False,
        ret_im=False,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a plane.

        Parameters
        ----------
        var : str
            The variable name
        resolution : float
            The resolution in m
        xmin : float
            The min x coordinate, or None for automatic
        ymin : float
            The min y coordinate, or None for automatic
        xmax : float
            The max x coordinate, or None for automatic
        ymax : float
            The max y coordinate, or None for automatic
        xlabel : str
            The x axis label
        ylabel : str
            The y axis label
        z : float
            The z coordinate of the plane
        xspace : float
            The extra space in x direction, before and after wind farm
        yspace : float
            The extra space in y direction, before and after wind farm
        levels : int
            The number of levels for the contourf plot, or None for pure image
        var_min : float
            Minimum variable value
        var_max : float
            Maximum variable value
        figsize : tuple
            The figsize for plt.Figure
        normalize_xy : float, optional
            Divide x and y by this value
        normalize_var : float, optional
            Divide the variable by this value
        title : str, optional
            The title
        vlabel : str, optional
            The variable label
        fig : plt.Figure, optional
            The figure object
        ax : plt.Axes, optional
            The figure axes
        add_bar : bool, optional
            Add a color bar
        cmap : str, optional
            The colormap
        weight_turbine : int, optional
            Index of the turbine from which to take the weight
        verbosity : int, optional
            The verbosity level
        ret_state : bool, optional
            Flag for state index return
        ret_im : bool, optional
            Flag for image return
        kwargs : dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Yields
        ------
        fig : matplotlib.Figure
            The figure object
        si : int, optional
            The state index
        im : matplotlib.collections.QuadMesh or matplotlib.QuadContourSet, optional
            The image object

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
        point_results = self.algo.calc_points(
            self.fres, points=g_pts, verbosity=verbosity, **kwargs
        )
        data = point_results[var].to_numpy()
        del point_results

        # take mean over states:
        weights = self.fres[FV.WEIGHT][:, weight_turbine].to_numpy()
        data = np.einsum("s,sp->p", weights, data)

        # find data min max:
        vmin = var_min if var_min is not None else np.min(data)
        vmax = var_max if var_max is not None else np.max(data)
        if normalize_var is not None:
            vmin /= normalize_var
            vmax /= normalize_var

        # normalize x and y:
        if normalize_xy is not None:
            x_pos /= normalize_xy
            y_pos /= normalize_xy

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
        )

        return out

    def get_mean_fig_vertical(
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
        verbosity=1,
        ret_state=False,
        ret_im=False,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a plane.

        Parameters
        ----------
        var : str
            The variable name
        resolution : float
            The resolution in m
        x_direction : float
            The direction of the x axis, 0 = north
        xmin : float
            The min x coordinate, or None for automatic
        zmin : float
            The min z coordinate
        xmax : float
            The max x coordinate, or None for automatic
        zmax : float
            The max z coordinate, or None for automatic
        xlabel : str
            The x axis label
        zlabel : str
            The z axis label
        y : float
            The y coordinate of the plane
        xspace : float
            The extra space in x direction, before and after wind farm
        zspace : float
            The extra space in z direction, below and above wind farm
        levels : int
            The number of levels for the contourf plot, or None for pure image
        var_min : float
            Minimum variable value
        var_max : float
            Maximum variable value
        figsize : tuple
            The figsize for plt.Figure
        normalize_x : float, optional
            Divide x by this value
        normalize_z : float, optional
            Divide z by this value
        normalize_var : float, optional
            Divide the variable by this value
        title : str, optional
            The title
        vlabel : str, optional
            The variable label
        fig : plt.Figure, optional
            The figure object
        ax : plt.Axes, optional
            The figure axes
        add_bar : bool, optional
            Add a color bar
        cmap : str, optional
            The colormap
        weight_turbine : int, optional
            Index of the turbine from which to take the weight
        verbosity : int, optional
            The verbosity level
        ret_state : bool, optional
            Flag for state index return
        ret_im : bool, optional
            Flag for image return
        kwargs : dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Yields
        ------
        fig : matplotlib.Figure
            The figure object
        si : int, optional
            The state index
        im : matplotlib.collections.QuadMesh or matplotlib.QuadContourSet, optional
            The image object

        """

        # prepare:
        n_states = self.algo.n_states
        n_turbines = self.algo.n_turbines
        n_x = np.append(wd2wdvec(x_direction), [0.0], axis=0)
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
        point_results = self.algo.calc_points(
            self.fres, points=g_pts, verbosity=verbosity, **kwargs
        )
        data = point_results[var].to_numpy()
        del point_results

        # take mean over states:
        weights = self.fres[FV.WEIGHT][:, weight_turbine].to_numpy()
        data = np.einsum("s,sp->p", weights, data)

        # find data min max:
        vmin = var_min if var_min is not None else np.min(data)
        vmax = var_max if var_max is not None else np.max(data)
        if normalize_var is not None:
            vmin /= normalize_var
            vmax /= normalize_var

        # normalize x and z:
        if normalize_x is not None:
            x_pos /= normalize_x
        if normalize_z is not None:
            z_pos /= normalize_z

        if title is None:
            title = f"States mean, x direction {x_direction}°"

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
        )

        return out

    def gen_states_fig_horizontal(
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
        verbosity=1,
        ret_state=False,
        ret_im=False,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a plane.

        Parameters
        ----------
        var : str
            The variable name
        resolution : float
            The resolution in m
        xmin : float
            The min x coordinate, or None for automatic
        ymin : float
            The min y coordinate, or None for automatic
        xmax : float
            The max x coordinate, or None for automatic
        ymax : float
            The max y coordinate, or None for automatic
        xlabel : str
            The x axis label
        ylabel : str
            The y axis label
        z : float
            The z coordinate of the plane
        xspace : float
            The extra space in x direction, before and after wind farm
        yspace : float
            The extra space in y direction, before and after wind farm
        levels : int
            The number of levels for the contourf plot, or None for pure image
        var_min : float
            Minimum variable value
        var_max : float
            Maximum variable value
        figsize : tuple
            The figsize for plt.Figure
        normalize_xy : float, optional
            Divide x and y by this value
        normalize_var : float, optional
            Divide the variable by this value
        title : str, optional
            The title
        vlabel : str, optional
            The variable label
        fig : plt.Figure, optional
            The figure object
        ax : plt.Axes, optional
            The figure axes
        add_bar : bool, optional
            Add a color bar
        cmap : str, optional
            The colormap
        quiver_n : int, optional
            Place a vector at each `n`th point
        quiver_pars : dict, optional
            Parameters for plt.quiver
        verbosity : int, optional
            The verbosity level
        ret_state : bool, optional
            Flag for state index return
        ret_im : bool, optional
            Flag for image return
        kwargs : dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Yields
        ------
        fig : matplotlib.Figure
            The figure object
        si : int, optional
            The state index
        im : matplotlib.collections.QuadMesh or matplotlib.QuadContourSet, optional
            The image object

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
        point_results = self.algo.calc_points(
            self.fres, points=g_pts, verbosity=verbosity, **kwargs
        )
        data = point_results[var].values
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
        vmin = var_min if var_min is not None else np.min(data)
        vmax = var_max if var_max is not None else np.max(data)
        if normalize_var is not None:
            vmin /= normalize_var
            vmax /= normalize_var

        # normalize x and y:
        if normalize_xy is not None:
            x_pos /= normalize_xy
            y_pos /= normalize_xy

        # loop over states:
        for si, s in enumerate(self.fres[FV.STATE].to_numpy()):

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
                title,
                add_bar,
                vlabel,
                ret_state,
                ret_im,
                quiv,
            )

            yield out

    def gen_states_fig_vertical(
        self,
        var,
        resolution,
        x_direction,
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
        verbosity=1,
        ret_state=False,
        ret_im=False,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a plane.

        Parameters
        ----------
        var : str
            The variable name
        resolution : float
            The resolution in m
        x_direction : float
            The direction of the x axis, 0 = north
        xmin : float
            The min x coordinate, or None for automatic
        zmin : float
            The min z coordinate
        xmax : float
            The max x coordinate, or None for automatic
        zmax : float
            The max z coordinate, or None for automatic
        xlabel : str
            The x axis label
        zlabel : str
            The z axis label
        y : float
            The y coordinate of the plane
        xspace : float
            The extra space in x direction, before and after wind farm
        zspace : float
            The extra space in z direction, below and above wind farm
        levels : int
            The number of levels for the contourf plot, or None for pure image
        var_min : float
            Minimum variable value
        var_max : float
            Maximum variable value
        figsize : tuple
            The figsize for plt.Figure
        normalize_x : float, optional
            Divide x by this value
        normalize_z : float, optional
            Divide z by this value
        normalize_var : float, optional
            Divide the variable by this value
        title : str, optional
            The title
        vlabel : str, optional
            The variable label
        fig : plt.Figure, optional
            The figure object
        ax : plt.Axes, optional
            The figure axes
        add_bar : bool, optional
            Add a color bar
        cmap : str, optional
            The colormap
        quiver_n : int, optional
            Place a vector at ech `n`th point
        quiver_pars : dict, optional
            Parameters for plt.quiver
        verbosity : int, optional
            The verbosity level
        ret_state : bool, optional
            Flag for state index return
        ret_im : bool, optional
            Flag for image return
        kwargs : dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Yields
        ------
        fig : matplotlib.Figure
            The figure object
        si : int, optional
            The state index
        im : matplotlib.collections.QuadMesh or matplotlib.QuadContourSet, optional
            The image object

        """

        # prepare:
        n_states = self.algo.n_states
        n_turbines = self.algo.n_turbines
        n_x = np.append(wd2wdvec(x_direction), [0.0], axis=0)
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
        point_results = self.algo.calc_points(
            self.fres, points=g_pts, verbosity=verbosity, **kwargs
        )
        data = point_results[var].values
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
        vmin = var_min if var_min is not None else np.min(data)
        vmax = var_max if var_max is not None else np.max(data)
        if normalize_var is not None:
            vmin /= normalize_var
            vmax /= normalize_var

        # normalize x and z:
        if normalize_x is not None:
            x_pos /= normalize_x
        if normalize_z is not None:
            z_pos /= normalize_z

        # loop over states:
        for si, s in enumerate(self.fres[FV.STATE].to_numpy()):

            ttl = f"State {s}" if title is None else title
            ttl += f", x direction {x_direction}°"

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
            )

            yield out
