import numpy as np

from foxes.output import Output
import foxes.constants as FC
import foxes.variables as FV

from .common import (
    get_grid_xy,
    get_grid_xz,
    get_grid_yz,
    calc_point_results,
    get_fig,
)


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

    :group: output.flow_plots_2d

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
        vmin=None,
        vmax=None,
        figsize=None,
        normalize_xy=None,
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
        Generates 2D farm flow figure in a horizontal xy-plane.

        Parameters
        ----------
        var: str
            The variable name
        resolution: float
            The resolution in m
        xmin: float, optional
            The min x coordinate, or None for automatic
        ymin: float, optional
            The min y coordinate, or None for automatic
        xmax: float, optional
            The max x coordinate, or None for automatic
        ymax: float, optional
            The max y coordinate, or None for automatic
        xlabel: str, optional
            The x axis label
        ylabel: str, optional
            The y axis label
        z: float, optional
            The z coordinate of the plane
        xspace: float, optional
            The extra space in x direction, before and after wind farm
        yspace: float, optional
            The extra space in y direction, before and after wind farm
        levels: int, optional
            The number of levels for the contourf plot, or None for pure image
        vmin: float, optional
            Minimum variable value
        vmax: float, optional
            Maximum variable value
        figsize: tuple, optional
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

        # create grid:
        x_pos, y_pos, z_pos, g_pts = get_grid_xy(
            self.fres, resolution, xmin, ymin, xmax, ymax, z, xspace, yspace, verbosity
        )

        # calculate point results:
        data = calc_point_results(
            algo=self.algo,
            farm_results=self.fres,
            g_pts=g_pts,
            verbosity=verbosity,
            **kwargs,
        )[var].to_numpy()

        # take mean over states:
        weights = self.fres[FV.WEIGHT][:, weight_turbine].to_numpy()
        data = np.einsum("s,sp->p", weights, data)[None]

        # find data min max:
        vmin = vmin if vmin is not None else np.nanmin(data)
        vmax = vmax if vmax is not None else np.nanmax(data)
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
        return get_fig(
            var=var,
            fig=fig,
            figsize=figsize,
            ax=ax,
            data=data,
            si=0,
            s=None,
            normalize_var=normalize_var,
            levels=levels,
            x_pos=x_pos,
            y_pos=y_pos,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            add_bar=add_bar,
            vlabel=vlabel,
            ret_state=ret_state,
            ret_im=ret_im,
            animated=animated,
        )

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
        vmin=None,
        vmax=None,
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
        x_direction: float, optional
            The direction of the x axis, 0 = north
        xmin: float, optional
            The min x coordinate, or None for automatic
        zmin: float, optional
            The min z coordinate
        xmax: float, optional
            The max x coordinate, or None for automatic
        zmax: float, optional
            The max z coordinate, or None for automatic
        xlabel: str, optional
            The x axis label
        zlabel: str, optional
            The z axis label
        y: float, optional
            The y coordinate of the plane
        xspace: float, optional
            The extra space in x direction, before and after wind farm
        zspace: float, optional
            The extra space in z direction, below and above wind farm
        levels: int, optional
            The number of levels for the contourf plot, or None for pure image
        vmin: float, optional
            Minimum variable value
        vmax: float, optional
            Maximum variable value
        figsize: tuple, optional
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

        # create grid:
        x_pos, y_pos, z_pos, g_pts = get_grid_xz(
            self.fres,
            resolution,
            x_direction,
            xmin,
            zmin,
            xmax,
            zmax,
            y,
            xspace,
            zspace,
            verbosity,
        )

        # calculate point results:
        data = calc_point_results(
            algo=self.algo,
            farm_results=self.fres,
            g_pts=g_pts,
            verbosity=verbosity,
            **kwargs,
        )[var].to_numpy()

        # take mean over states:
        weights = self.fres[FV.WEIGHT][:, weight_turbine].to_numpy()
        data = np.einsum("s,sp->p", weights, data)[None]

        # find data min max:
        vmin = vmin if vmin is not None else np.nanmin(data)
        vmax = vmax if vmax is not None else np.nanmax(data)
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
        return get_fig(
            var=var,
            fig=fig,
            figsize=figsize,
            ax=ax,
            data=data,
            si=0,
            s=None,
            normalize_var=normalize_var,
            levels=levels,
            x_pos=x_pos,
            y_pos=z_pos,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=zlabel,
            title=title,
            add_bar=add_bar,
            vlabel=vlabel,
            ret_state=ret_state,
            ret_im=ret_im,
            animated=animated,
        )

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
        vmin=None,
        vmax=None,
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
        x_direction: float, optional
            The direction of the x axis, 0 = north
        ymin: float, optional
            The min y coordinate, or None for automatic
        zmin: float, optional
            The min z coordinate
        ymax: float, optional
            The max y coordinate, or None for automatic
        zmax: float, optional
            The max z coordinate, or None for automatic
        ylabel: str, optional
            The y axis label
        zlabel: str, optional
            The z axis label
        x: float, optional
            The x coordinate of the plane
        yspace: float, optional
            The extra space in y direction, before and after wind farm
        zspace: float, optional
            The extra space in z direction, below and above wind farm
        levels: int, optional
            The number of levels for the contourf plot, or None for pure image
        vmin: float, optional
            Minimum variable value
        vmax: float, optional
            Maximum variable value
        figsize: tuple, optional
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

        # create grid:
        x_pos, y_pos, z_pos, g_pts = get_grid_yz(
            self.fres,
            resolution,
            x_direction,
            ymin,
            zmin,
            ymax,
            zmax,
            x,
            yspace,
            zspace,
            verbosity,
        )

        # calculate point results:
        data = calc_point_results(
            algo=self.algo,
            farm_results=self.fres,
            g_pts=g_pts,
            verbosity=verbosity,
            **kwargs,
        )[var].to_numpy()

        # take mean over states:
        weights = self.fres[FV.WEIGHT][:, weight_turbine].to_numpy()
        data = np.einsum("s,sp->p", weights, data)[None]

        # find data min max:
        vmin = vmin if vmin is not None else np.nanmin(data)
        vmax = vmax if vmax is not None else np.nanmax(data)
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
        return get_fig(
            var=var,
            fig=fig,
            figsize=figsize,
            ax=ax,
            data=data,
            si=0,
            s=None,
            normalize_var=normalize_var,
            levels=levels,
            x_pos=y_pos,
            y_pos=z_pos,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            xlabel=ylabel,
            ylabel=zlabel,
            title=title,
            add_bar=add_bar,
            vlabel=vlabel,
            ret_state=ret_state,
            ret_im=ret_im,
            invert_axis="x",
            animated=animated,
        )

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
        vmin=None,
        vmax=None,
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
        xmin: float, optional
            The min x coordinate, or None for automatic
        ymin: float, optional
            The min y coordinate, or None for automatic
        xmax: float, optional
            The max x coordinate, or None for automatic
        ymax: float, optional
            The max y coordinate, or None for automatic
        xlabel: str, optional
            The x axis label
        ylabel: str, optional
            The y axis label
        z: float, optional
            The z coordinate of the plane
        xspace: float, optional
            The extra space in x direction, before and after wind farm
        yspace: float, optional
            The extra space in y direction, before and after wind farm
        levels: int, optional
            The number of levels for the contourf plot, or None for pure image
        vmin: float, optional
            Minimum variable value
        vmax: float, optional
            Maximum variable value
        figsize: tuple, optional
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

        # create grid:
        x_pos, y_pos, z_pos, g_pts = get_grid_xy(
            self.fres, resolution, xmin, ymin, xmax, ymax, z, xspace, yspace, verbosity
        )

        # calculate point results:
        point_results = calc_point_results(
            algo=self.algo,
            farm_results=self.fres,
            g_pts=g_pts,
            verbosity=verbosity,
            **kwargs,
        )
        data = point_results[var].to_numpy()

        # define wind vector arrows:
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
        qpars.update(quiver_pars)
        quiv = (
            None
            if quiver_n is None
            else (
                quiver_n,
                qpars,
                point_results[FV.WD].to_numpy(),
                point_results[FV.WS].to_numpy(),
            )
        )
        del point_results

        # find data min max:
        vmin = vmin if vmin is not None else np.nanmin(data)
        vmax = vmax if vmax is not None else np.nanmax(data)
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

            out = get_fig(
                var=var,
                fig=fig,
                figsize=figsize,
                ax=ax,
                data=data,
                si=si,
                s=s,
                normalize_var=normalize_var,
                levels=levels,
                x_pos=x_pos,
                y_pos=y_pos,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                xlabel=xlabel,
                ylabel=ylabel,
                title=ttl,
                add_bar=add_bar,
                vlabel=vlabel,
                ret_state=ret_state,
                ret_im=ret_im,
                quiv=quiv,
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
        vmin=None,
        vmax=None,
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
        x_direction: float, optional
            The direction of the x axis, 0 = north
        xmin: float, optional
            The min x coordinate, or None for automatic
        zmin: float, optional
            The min z coordinate
        xmax: float, optional
            The max x coordinate, or None for automatic
        zmax: float, optional
            The max z coordinate, or None for automatic
        xlabel: str, optional
            The x axis label
        zlabel: str, optional
            The z axis label
        y: float, optional
            The y coordinate of the plane
        xspace: float, optional
            The extra space in x direction, before and after wind farm
        zspace: float, optional
            The extra space in z direction, below and above wind farm
        levels: int, optional
            The number of levels for the contourf plot, or None for pure image
        vmin: float, optional
            Minimum variable value
        vmax: float, optional
            Maximum variable value
        figsize: tuple, optional
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

        # create grid:
        x_pos, y_pos, z_pos, g_pts = get_grid_xz(
            self.fres,
            resolution,
            x_direction,
            xmin,
            zmin,
            xmax,
            zmax,
            y,
            xspace,
            zspace,
            verbosity,
        )

        # calculate point results:
        point_results = calc_point_results(
            algo=self.algo,
            farm_results=self.fres,
            g_pts=g_pts,
            verbosity=verbosity,
            **kwargs,
        )
        data = point_results[var].to_numpy()

        # define wind vector arrows:
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
        qpars.update(quiver_pars)
        quiv = (
            None
            if quiver_n is None
            else (
                quiver_n,
                qpars,
                point_results[FV.WD].to_numpy(),
                point_results[FV.WS].to_numpy(),
            )
        )
        del point_results

        # find data min max:
        vmin = vmin if vmin is not None else np.nanmin(data)
        vmax = vmax if vmax is not None else np.nanmax(data)
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

            out = get_fig(
                var=var,
                fig=fig,
                figsize=figsize,
                ax=ax,
                data=data,
                si=si,
                s=s,
                normalize_var=normalize_var,
                levels=levels,
                x_pos=x_pos,
                y_pos=z_pos,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                xlabel=xlabel,
                ylabel=zlabel,
                title=ttl,
                add_bar=add_bar,
                vlabel=vlabel,
                ret_state=ret_state,
                ret_im=ret_im,
                quiv=quiv,
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
        vmin=None,
        vmax=None,
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
        x_direction: float, optional
            The direction of the x axis, 0 = north
        ymin: float, optional
            The min y coordinate, or None for automatic
        zmin: float, optional
            The min z coordinate
        ymax: float, optional
            The max y coordinate, or None for automatic
        zmax: float, optional
            The max z coordinate, or None for automatic
        ylabel: str, optional
            The y axis label
        zlabel: str, optional
            The z axis label
        x: float, optional
            The x coordinate of the plane
        yspace: float, optional
            The extra space in y direction, left and right of wind farm
        zspace: float, optional
            The extra space in z direction, below and above wind farm
        levels: int, optional
            The number of levels for the contourf plot, or None for pure image
        vmin: float, optional
            Minimum variable value
        vmax: float, optional
            Maximum variable value
        figsize: tuple, optional
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

        # create grid:
        x_pos, y_pos, z_pos, g_pts = get_grid_yz(
            self.fres,
            resolution,
            x_direction,
            ymin,
            zmin,
            ymax,
            zmax,
            x,
            yspace,
            zspace,
            verbosity,
        )

        # calculate point results:
        point_results = calc_point_results(
            algo=self.algo,
            farm_results=self.fres,
            g_pts=g_pts,
            verbosity=verbosity,
            **kwargs,
        )
        data = point_results[var].to_numpy()
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
        qpars.update(quiver_pars)
        quiv = (
            None
            if quiver_n is None
            else (
                quiver_n,
                qpars,
                point_results[FV.WD].to_numpy(),
                point_results[FV.WS].to_numpy(),
            )
        )
        del point_results

        # find data min max:
        vmin = vmin if vmin is not None else np.nanmin(data)
        vmax = vmax if vmax is not None else np.nanmax(data)
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

            out = get_fig(
                var=var,
                fig=fig,
                figsize=figsize,
                ax=ax,
                data=data,
                si=si,
                s=s,
                normalize_var=normalize_var,
                levels=levels,
                x_pos=y_pos,
                y_pos=z_pos,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                xlabel=ylabel,
                ylabel=zlabel,
                title=ttl,
                add_bar=add_bar,
                vlabel=vlabel,
                ret_state=ret_state,
                ret_im=ret_im,
                quiv=quiv,
                invert_axis="x",
                animated=animated,
            )

            yield out
