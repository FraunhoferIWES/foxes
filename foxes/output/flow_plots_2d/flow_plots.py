from __future__ import annotations

import numpy as np
from typing import Any, Iterator, TYPE_CHECKING
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from foxes.core import TData
from foxes.output import SliceData
import foxes.variables as FV
import foxes.constants as FC

from .get_fig import get_fig
from ..grids import get_grid_xy, np2np_sp

if TYPE_CHECKING:
    from foxes.core import FData, MData


class FlowPlots2D(SliceData):
    """
    Class for horizontal or vertical 2D flow plots

    :group: output.flow_plots_2d

    """

    def get_mean_data_xy(  # type: ignore[override]
        self,
        var: str,
        vmin: float | None = None,
        vmax: float | None = None,
        data_format: str = "numpy",
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Any, Any]:
        """
        Generates 2D farm flow figure in a horizontal xy-plane.

        Parameters
        ----------
        var
            The variable name
        x_direction
            The direction of the x axis, 0 = north
        vmin
            The minimal variable value
        vmax
            The maximal variable value
        data_format
            The output data format: numpy, pandas, xarray
        kwargs
            Additional parameters for SliceData.get_mean_data_xy

        Returns
        -------
        parameters
            The parameters used
        data
            The gridded data
        grid_data
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        variables = list(set([var] + [FV.WD, FV.WS]))

        data, gdata = super().get_mean_data_xy(
            variables=variables,
            vmin={var: vmin} if vmin is not None else {},
            vmax={var: vmax} if vmax is not None else {},
            data_format=data_format,
            ret_grid=True,
            ret_states=False,
            **kwargs,
        )

        parameters = dict(
            var=var,
            variables=variables,
            vmin=vmin,
            vmax=vmax,
            data_format=data_format,
        )

        return parameters, data, gdata

    def get_mean_data_yz(  # type: ignore[override]
        self,
        var: str,
        x_direction: float = 270.0,
        vmin: float | None = None,
        vmax: float | None = None,
        data_format: str = "numpy",
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Any, Any]:
        """
        Generates 2D farm flow figure in a horizontal yz-plane.

        Parameters
        ----------
        var
            The variable name
        x_direction
            The direction of the x axis, 0 = north
        vmin
            The minimal variable value
        vmax
            The maximal variable value
        data_format
            The output data format: numpy, pandas, xarray
        kwargs
            Additional parameters for SliceData.get_mean_data_yz

        Returns
        -------
        parameters
            The parameters used
        data
            The gridded data
        grid_data
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        variables = list(set([var] + [FV.WD, FV.WS]))

        data, gdata = super().get_mean_data_yz(
            variables=variables,
            x_direction=x_direction,
            vmin={var: vmin} if vmin is not None else {},
            vmax={var: vmax} if vmax is not None else {},
            data_format=data_format,
            ret_grid=True,
            ret_states=False,
            **kwargs,
        )

        parameters = dict(
            var=var,
            x_direction=x_direction,
            variables=variables,
            vmin=vmin,
            vmax=vmax,
            data_format=data_format,
        )

        return parameters, data, gdata

    def get_mean_data_xz(  # type: ignore[override]
        self,
        var: str,
        x_direction: float = 270.0,
        vmin: float | None = None,
        vmax: float | None = None,
        data_format: str = "numpy",
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Any, Any]:
        """
        Generates 2D farm flow figure in a horizontal xz-plane.

        Parameters
        ----------
        var
            The variable name
        x_direction
            The direction of the x axis, 0 = north
        vmin
            The minimal variable value
        vmax
            The maximal variable value
        data_format
            The output data format: numpy, pandas, xarray
        kwargs
            Additional parameters for SliceData.get_mean_data_xz

        Returns
        -------
        parameters
            The parameters used
        data
            The gridded data
        grid_data
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        variables = list(set([var] + [FV.WD, FV.WS]))

        data, gdata = super().get_mean_data_xz(
            variables=variables,
            x_direction=x_direction,
            vmin={var: vmin} if vmin is not None else {},
            vmax={var: vmax} if vmax is not None else {},
            data_format=data_format,
            ret_grid=True,
            ret_states=False,
            **kwargs,
        )

        parameters = dict(
            var=var,
            x_direction=x_direction,
            variables=variables,
            vmin=vmin,
            vmax=vmax,
            data_format=data_format,
        )

        return parameters, data, gdata

    def get_mean_fig_xy(
        self,
        mean_data_xy: tuple[Any, Any, Any],
        xlabel: str = "x [m]",
        ylabel: str = "y [m]",
        levels: int | None = None,
        figsize: tuple[int, int] | None = None,
        title: Any = None,
        vlabel: str | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
        add_bar: bool = True,
        cmap: str | None = None,
        quiver_n: int | None = None,
        quiver_pars: dict[str, Any] | None = None,
        ret_state: bool = False,
        ret_im: bool = False,
        ret_data: bool = False,
        animated: bool = False,
    ) -> Any:
        """
        Generates 2D farm flow figure in a horizontal xy-plane.

        Parameters
        ----------
        mean_data_xy
            The pre-calculated data from get_mean_data_xy,
            (parameters, data, grid_data)
        xlabel
            The x axis label
        ylabel
            The y axis label
        levels
            The number of levels for the contourf plot, or None for pure image
        figsize
            The figsize for plt.Figure
        title
            The title
        vlabel
            The variable label
        fig
            The figure object
        ax
            The figure axes
        add_bar
            Add a color bar
        cmap
            The colormap
        quiver_n
            Place a vector at each `n`th point
        quiver_pars
            Parameters for plt.quiver
        ret_state
            Flag for state index return
        ret_im
            Flag for image return
        ret_data
            Flag for returning image data
        animated
            Switch for usage for an animation

        Returns
        -------
        fig
            The figure object
        si
            The state index
        im
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet
        data
            The image data, shape: (n_x, n_y)

        """
        # read data:
        parameters, data, gdata = mean_data_xy
        var = parameters["var"]
        variables = parameters["variables"]
        vmin = parameters["vmin"]
        vmax = parameters["vmax"]
        data_format = parameters["data_format"]
        vi = variables.index(var)
        wdi = variables.index(FV.WD)
        wsi = variables.index(FV.WS)
        x_pos, y_pos, z_pos, __ = gdata

        if data_format != "numpy":
            raise NotImplementedError(
                f"Only numpy data_format is supported here, got {data_format}"
            )

        if title is None:
            title = f"States mean, z =  {int(np.round(z_pos))} m"

        # define wind vector arrows:
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
        quiver_pars = {} if quiver_pars is None else quiver_pars
        qpars.update(quiver_pars)
        quiv = (
            None
            if quiver_n is None
            else (
                quiver_n,
                qpars,
                data[None, :, :, wdi],
                data[None, :, :, wsi],
            )
        )

        # create plot:
        out = get_fig(
            var=var,
            fig=fig,
            figsize=figsize,
            ax=ax,
            data=data[None, :, :, vi],
            si=0,
            s=None,
            levels=levels,
            x_pos=x_pos,
            y_pos=y_pos,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            add_bar=add_bar,
            vlabel=vlabel,
            quiv=quiv,
            vmin=vmin,
            vmax=vmax,
            ret_state=ret_state,
            ret_im=ret_im,
            animated=animated,
        )

        if ret_data:
            out = list(out) if isinstance(out, tuple) else [out]
            return tuple(out + [data[:, :, 0]])

        return out

    def get_mean_fig_xz(
        self,
        mean_data_xz: tuple[Any, Any, Any],
        xlabel: str = "x [m]",
        zlabel: str = "z [m]",
        levels: int | None = None,
        figsize: tuple[int, int] | None = None,
        title: Any = None,
        vlabel: str | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
        add_bar: bool = True,
        cmap: str | None = None,
        quiver_n: int | None = None,
        quiver_pars: dict[str, Any] | None = None,
        ret_state: bool = False,
        ret_im: bool = False,
        ret_data: bool = False,
        animated: bool = False,
    ) -> Any:
        """
        Generates 2D farm flow figure in a horizontal xz-plane.

        Parameters
        ----------
        mean_data_xz
            The pre-calculated data from get_mean_data_xz,
            (parameters, data, grid_data)
        xlabel
            The x axis label
        zlabel
            The z axis label
        levels
            The number of levels for the contourf plot, or None for pure image
        figsize
            The figsize for plt.Figure
        title
            The title
        vlabel
            The variable label
        fig
            The figure object
        ax
            The figure axes
        add_bar
            Add a color bar
        cmap
            The colormap
        quiver_n
            Place a vector at each `n`th point
        quiver_pars
            Parameters for plt.quiver
        ret_state
            Flag for state index return
        ret_im
            Flag for image return
        ret_data
            Flag for returning image data
        animated
            Switch for usage for an animation

        Returns
        -------
        fig
            The figure object
        si
            The state index
        im
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet
        data
            The image data, shape: (n_x, n_y)

        """
        if self.nofig:
            return None

        # read data:
        parameters, data, gdata = mean_data_xz
        var = parameters["var"]
        variables = parameters["variables"]
        vmin = parameters["vmin"]
        vmax = parameters["vmax"]
        x_direction = parameters["x_direction"]
        data_format = parameters["data_format"]
        vi = variables.index(var)
        wdi = variables.index(FV.WD)
        wsi = variables.index(FV.WS)
        x_pos, y_pos, z_pos, __ = gdata

        if data_format != "numpy":
            raise NotImplementedError(
                f"Only numpy data_format is supported here, got {data_format}"
            )

        if title is None:
            title = f"States mean, x direction {x_direction}°, y =  {int(np.round(y_pos))} m"

        # define wind vector arrows:
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
        quiver_pars = {} if quiver_pars is None else quiver_pars
        qpars.update(quiver_pars)
        quiv = (
            None
            if quiver_n is None
            else (
                quiver_n,
                qpars,
                data[None, :, :, wdi],
                data[None, :, :, wsi],
            )
        )

        # create plot:
        out = get_fig(
            var=var,
            fig=fig,
            figsize=figsize,
            ax=ax,
            data=data[None, :, :, vi],
            si=0,
            s=None,
            levels=levels,
            x_pos=x_pos,
            y_pos=z_pos,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=zlabel,
            title=title,
            add_bar=add_bar,
            vlabel=vlabel,
            vmin=vmin,
            vmax=vmax,
            ret_state=ret_state,
            ret_im=ret_im,
            quiv=quiv,
            animated=animated,
        )

        if ret_data:
            out = list(out) if isinstance(out, tuple) else [out]
            return tuple(out + [data[:, :, 0]])

        return out

    def get_mean_fig_yz(
        self,
        mean_data_yz: tuple[Any, Any, Any],
        ylabel: str = "x [m]",
        zlabel: str = "z [m]",
        levels: int | None = None,
        figsize: tuple[int, int] | None = None,
        title: Any = None,
        vlabel: str | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
        add_bar: bool = True,
        cmap: str | None = None,
        quiver_n: int | None = None,
        quiver_pars: dict[str, Any] | None = None,
        ret_state: bool = False,
        ret_im: bool = False,
        ret_data: bool = False,
        animated: bool = False,
    ) -> Any:
        """
        Generates 2D farm flow figure in a horizontal yz-plane.

        Parameters
        ----------
        mean_data_yz
            The pre-calculated data from get_mean_data_yz,
            (parameters, data, grid_data)
        x_direction
            The direction of the x axis, 0 = north
        ylabel
            The y axis label
        zlabel
            The z axis label
        levels
            The number of levels for the contourf plot, or None for pure image
        figsize
            The figsize for plt.Figure
        title
            The title
        vlabel
            The variable label
        fig
            The figure object
        ax
            The figure axes
        add_bar
            Add a color bar
        cmap
            The colormap
        quiver_n
            Place a vector at each `n`th point
        quiver_pars
            Parameters for plt.quiver
        ret_state
            Flag for state index return
        ret_im
            Flag for image return
        ret_data
            Flag for returning image data
        animated
            Switch for usage for an animation

        Returns
        -------
        fig
            The figure object
        si
            The state index
        im
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet
        data
            The image data, shape: (n_x, n_y)

        """
        if self.nofig:
            return None

        # read data:
        parameters, data, gdata = mean_data_yz
        var = parameters["var"]
        variables = parameters["variables"]
        vmin = parameters["vmin"]
        vmax = parameters["vmax"]
        x_direction = parameters["x_direction"]
        data_format = parameters["data_format"]
        vi = variables.index(var)
        wdi = variables.index(FV.WD)
        wsi = variables.index(FV.WS)
        x_pos, y_pos, z_pos, __ = gdata

        if data_format != "numpy":
            raise NotImplementedError(
                f"Only numpy data_format is supported here, got {data_format}"
            )

        if title is None:
            title = f"States mean, x direction {x_direction}°, x =  {int(np.round(x_pos))} m"

        # define wind vector arrows:
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
        quiver_pars = {} if quiver_pars is None else quiver_pars
        qpars.update(quiver_pars)
        quiv = (
            None
            if quiver_n is None
            else (
                quiver_n,
                qpars,
                data[None, :, :, wdi],
                data[None, :, :, wsi],
            )
        )

        # create plot:
        out = get_fig(
            var=var,
            fig=fig,
            figsize=figsize,
            ax=ax,
            data=data[None, :, :, vi],
            si=0,
            s=None,
            levels=levels,
            x_pos=y_pos,
            y_pos=z_pos,
            cmap=cmap,
            xlabel=ylabel,
            ylabel=zlabel,
            title=title,
            add_bar=add_bar,
            vlabel=vlabel,
            vmin=vmin,
            vmax=vmax,
            ret_state=ret_state,
            ret_im=ret_im,
            quiv=quiv,
            invert_axis="x",
            animated=animated,
        )

        if ret_data:
            out = list(out) if isinstance(out, tuple) else [out]
            return tuple(out + [data[:, :, 0]])

        return out

    def get_states_data_xy(  # type: ignore[override]
        self,
        var: str,
        vmin: float | None = None,
        vmax: float | None = None,
        data_format: str = "numpy",
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Any, Any, Any]:
        """
        Generates 2D farm flow figure in a horizontal xy-plane.

        Parameters
        ----------
        var
            The variable name
        vmin
            The minimal variable value
        vmax
            The maximal variable value
        data_format
            The output data format: numpy, pandas, xarray
        kwargs
            Additional parameters for SliceData.get_states_data_xy

        Returns
        -------
        parameters
            The parameters used
        data
            The gridded data
        states
            The states indices
        grid_data
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        variables = list(set([var] + [FV.WD, FV.WS]))

        data, states, gdata = super().get_states_data_xy(
            variables=variables,
            vmin={var: vmin} if vmin is not None else {},
            vmax={var: vmax} if vmax is not None else {},
            data_format=data_format,
            ret_states=True,
            ret_grid=True,
            **kwargs,
        )

        pars = dict(
            var=var,
            variables=variables,
            vmin=vmin,
            vmax=vmax,
            data_format=data_format,
        )

        return pars, data, states, gdata

    def get_states_data_xz(  # type: ignore[override]
        self,
        var: str,
        x_direction: float = 270.0,
        vmin: float | None = None,
        vmax: float | None = None,
        data_format: str = "numpy",
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Any, Any, Any]:
        """
        Generates 2D farm flow figure in a horizontal xz-plane.

        Parameters
        ----------
        var
            The variable name
        x_direction
            The direction of the x axis, 0 = north
        vmin
            The minimal variable value
        vmax
            The maximal variable value
        data_format
            The output data format: numpy, pandas, xarray
        kwargs
            Additional parameters for SliceData.get_states_data_xz

        Returns
        -------
        parameters
            The parameters used
        data
            The gridded data
        states
            The states indices
        grid_data
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        variables = list(set([var] + [FV.WD, FV.WS]))

        data, states, gdata = super().get_states_data_xz(
            variables=variables,
            x_direction=x_direction,
            vmin={var: vmin} if vmin is not None else {},
            vmax={var: vmax} if vmax is not None else {},
            data_format=data_format,
            ret_states=True,
            ret_grid=True,
            **kwargs,
        )

        pars = dict(
            var=var,
            x_direction=x_direction,
            variables=variables,
            vmin=vmin,
            vmax=vmax,
            data_format=data_format,
        )

        return pars, data, states, gdata

    def get_states_data_yz(  # type: ignore[override]
        self,
        var: str,
        x_direction: float = 270.0,
        vmin: float | None = None,
        vmax: float | None = None,
        data_format: str = "numpy",
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Any, Any, Any]:
        """
        Generates 2D farm flow figure in a horizontal yz-plane.

        Parameters
        ----------
        var
            The variable name
        x_direction
            The direction of the x axis, 0 = north
        vmin
            The minimal variable value
        vmax
            The maximal variable value
        data_format
            The output data format: numpy, pandas, xarray
        kwargs
            Additional parameters for SliceData.get_states_data_yz

        Returns
        -------
        parameters
            The parameters used
        data
            The gridded data
        states
            The states indices
        grid_data
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        variables = list(set([var] + [FV.WD, FV.WS]))

        data, states, gdata = super().get_states_data_yz(
            variables=variables,
            x_direction=x_direction,
            vmin={var: vmin} if vmin is not None else {},
            vmax={var: vmax} if vmax is not None else {},
            data_format=data_format,
            ret_states=True,
            ret_grid=True,
            **kwargs,
        )

        pars = dict(
            var=var,
            x_direction=x_direction,
            variables=variables,
            vmin=vmin,
            vmax=vmax,
            data_format=data_format,
        )

        return pars, data, states, gdata

    def gen_states_fig_xy(
        self,
        states_data_xy: tuple[Any, Any, Any, Any],
        title: Any = None,
        add_bar: bool = True,
        quiver_n: int | None = None,
        quiver_pars: dict[str, Any] | None = None,
        animated: bool = False,
        rotor_color: Any = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """
        Generates 2D farm flow figure in a horizontal xy-plane.

        Parameters
        ----------
        states_data_xy
            The pre-calculated data from get_states_data_xy,
            (parameters, data, states, grid_data)
        title
            The title
        add_bar
            Add a color bar
        quiver_n
            Place a vector at each `n`th point
        quiver_pars
            Parameters for plt.quiver
        animated
            Switch for usage for an animation
        rotor_color
            Indicate the rotor orientation by a colored line
        kwargs
            Additional parameters for get_fig(), e.g.
            xlabel, ylabel, levels, figsize, vlabel,
            fig, ax, cmap, ret_state, ret_im

        Yields
        ------
        fig
            The figure object
        si
            The state index
        im
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet

        """
        if self.nofig:
            yield None

        # read data:
        parameters, data, states, gdata = states_data_xy
        var = parameters["var"]
        variables = parameters["variables"]
        vmin = parameters["vmin"]
        vmax = parameters["vmax"]
        data_format = parameters["data_format"]
        vi = variables.index(var)
        wdi = variables.index(FV.WD)
        wsi = variables.index(FV.WS)
        x_pos, y_pos, z_pos, __ = gdata

        if data_format != "numpy":
            raise NotImplementedError(
                f"Only numpy data_format is supported here, got {data_format}"
            )

        # define wind vector arrows:
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
        quiver_pars = {} if quiver_pars is None else quiver_pars
        qpars.update(quiver_pars)
        quiv = (
            None
            if quiver_n is None
            else (
                quiver_n,
                qpars,
                data[..., wdi],
                data[..., wsi],
            )
        )

        # loop over states:
        for si, s in enumerate(states):
            if animated and si == 0:
                vmin = vmin if vmin is not None else np.min(data[..., vi])
                vmax = vmax if vmax is not None else np.max(data[..., vi])
            elif animated and si > 0:
                add_bar = False

            if not animated and title is None:
                ttl = f"State {s}"
                ttl += f", z =  {int(np.round(z_pos))} m"
            elif callable(title):
                ttl = title(si, s)
            else:
                ttl = title

            # get data for show_turbines
            if rotor_color is not None:
                try:
                    turb_angle = self.fres[FV.YAW][si]
                except KeyError:
                    try:
                        turb_angle = self.fres[FV.AMB_WD][si] + self.fres[FV.YAWM][si]
                    except KeyError:
                        turb_angle = self.fres[FV.AMB_WD][si]

                show_rotor_dict = {
                    "color": rotor_color,
                    "D": self.fres[FV.D][si],
                    "H": self.fres[FV.H][si],
                    "X": self.fres[FV.X][si],
                    "Y": self.fres[FV.Y][si],
                    "AMB_WD": self.fres[FV.AMB_WD][si],
                    "turb_angle": turb_angle,
                }
            else:
                show_rotor_dict = None

            out = get_fig(
                var=var,
                data=data[..., vi],
                si=si,
                s=s,
                x_pos=x_pos,
                y_pos=y_pos,
                title=ttl,
                add_bar=add_bar,
                vmin=vmin,
                vmax=vmax,
                quiv=quiv,
                animated=animated,
                show_rotor_dict=show_rotor_dict,
                rotor_plane="xy",
                **kwargs,
            )

            yield out

    def gen_states_fig_xz(
        self,
        states_data_xz: tuple[Any, Any, Any, Any],
        title: Any = None,
        add_bar: bool = True,
        quiver_n: int | None = None,
        quiver_pars: dict[str, Any] | None = None,
        animated: bool = False,
        rotor_color: Any = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """
        Generates 2D farm flow figure in a vertical xz-plane.

        Parameters
        ----------
        states_data_xz
            The pre-calculated data from get_states_data_xz,
            (parameters, data, states, grid_data)
        title
            The title
        add_bar
            Add a color bar
        quiver_n
            Place a vector at ech `n`th point
        quiver_pars
            Parameters for plt.quiver
        animated
            Switch for usage for an animation
        rotor_color
            Indicate the rotor orientation by a colored line
        kwargs
            Additional parameters for get_fig(), e.g.
            xlabel, ylabel, levels, figsize, vlabel,
            fig, ax, cmap, ret_state, ret_im

        Yields
        ------
        fig
            The figure object
        si
            The state index
        im
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet

        """
        if self.nofig:
            yield None

        # read data:
        parameters, data, states, gdata = states_data_xz
        var = parameters["var"]
        variables = parameters["variables"]
        vmin = parameters["vmin"]
        vmax = parameters["vmax"]
        x_direction = parameters["x_direction"]
        data_format = parameters["data_format"]
        vi = variables.index(var)
        wdi = variables.index(FV.WD)
        wsi = variables.index(FV.WS)
        x_pos, y_pos, z_pos, __ = gdata

        if data_format != "numpy":
            raise NotImplementedError(
                f"Only numpy data_format is supported here, got {data_format}"
            )

        # define wind vector arrows:
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
        quiver_pars = {} if quiver_pars is None else quiver_pars
        qpars.update(quiver_pars)
        quiv = (
            None
            if quiver_n is None
            else (
                quiver_n,
                qpars,
                data[..., wdi],
                data[..., wsi],
            )
        )

        # loop over states:
        for si, s in enumerate(states):
            if animated and si > 0 and vmin is not None and vmax is not None:
                add_bar = False
            if not animated and title is None:
                ttl = f"State {s}"
                ttl += f", x direction = {x_direction}°"
                ttl += f", y =  {int(np.round(y_pos))} m"
            elif callable(title):
                ttl = title(si, s)
            else:
                ttl = title

            # get data for show_turbines
            if rotor_color is not None:
                try:
                    turb_angle = self.fres[FV.YAW][si]
                except KeyError:
                    try:
                        turb_angle = self.fres[FV.AMB_WD][si] + self.fres[FV.YAWM][si]
                    except KeyError:
                        turb_angle = self.fres[FV.AMB_WD][si]

                show_rotor_dict = {
                    "color": rotor_color,
                    "D": self.fres[FV.D][si],
                    "H": self.fres[FV.H][si],
                    "X": self.fres[FV.X][si],
                    "Y": self.fres[FV.Y][si],
                    "AMB_WD": self.fres[FV.AMB_WD][si],
                    "turb_angle": turb_angle,
                }
            else:
                show_rotor_dict = None

            out = get_fig(
                var=var,
                data=data[..., vi],
                si=si,
                s=s,
                x_pos=x_pos,
                y_pos=z_pos,
                title=ttl,
                add_bar=add_bar,
                quiv=quiv,
                vmin=vmin,
                vmax=vmax,
                animated=animated,
                show_rotor_dict=show_rotor_dict,
                rotor_plane="xz",
                rotor_slice={"axis": "y", "value": y_pos, "tol": 0.0},
                **kwargs,
            )

            yield out

    def gen_states_fig_yz(
        self,
        states_data_yz: tuple[Any, Any, Any, Any],
        title: Any = None,
        add_bar: bool = True,
        quiver_n: int | None = None,
        quiver_pars: dict[str, Any] | None = None,
        animated: bool = False,
        rotor_color: Any = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """
        Generates 2D farm flow figure in a vertical yz-plane.

        Parameters
        ----------
        states_data_yz
            The pre-calculated data from get_states_data_yz,
            (parameters, data, states, grid_data)
        title
            The title
        add_bar
            Add a color bar
        quiver_n
            Place a vector at ech `n`th point
        quiver_pars
            Parameters for plt.quiver
        animated
            Switch for usage for an animation
        rotor_color
            Indicate the rotor orientation by a colored line
        kwargs
            Additional parameters for get_fig(), e.g.
            xlabel, ylabel, levels, figsize, vlabel,
            fig, ax, cmap, ret_state, ret_im

        Yields
        ------
        fig
            The figure object
        si
            The state index
        im
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet

        """
        if self.nofig:
            yield None

        # read data:
        parameters, data, states, gdata = states_data_yz
        var = parameters["var"]
        variables = parameters["variables"]
        vmin = parameters["vmin"]
        vmax = parameters["vmax"]
        x_direction = parameters["x_direction"]
        data_format = parameters["data_format"]
        vi = variables.index(var)
        wdi = variables.index(FV.WD)
        wsi = variables.index(FV.WS)
        x_pos, y_pos, z_pos, __ = gdata

        if data_format != "numpy":
            raise NotImplementedError(
                f"Only numpy data_format is supported here, got {data_format}"
            )

        # define wind vector arrows:
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
        quiver_pars = {} if quiver_pars is None else quiver_pars
        qpars.update(quiver_pars)
        quiv = (
            None
            if quiver_n is None
            else (
                quiver_n,
                qpars,
                data[..., wdi],
                data[..., wsi],
            )
        )

        # loop over states:
        for si, s in enumerate(states):
            if animated and si > 0 and vmin is not None and vmax is not None:
                add_bar = False
            if not animated and title is None:
                ttl = f"State {s}" if title is None else title
                ttl += f", x direction = {x_direction}°"
                ttl += f", x =  {int(np.round(x_pos))} m"
            elif callable(title):
                ttl = title(si, s)
            else:
                ttl = title

            # get data for show_turbines
            if rotor_color is not None:
                try:
                    turb_angle = self.fres[FV.YAW][si]
                except KeyError:
                    try:
                        turb_angle = self.fres[FV.AMB_WD][si] + self.fres[FV.YAWM][si]
                    except KeyError:
                        turb_angle = self.fres[FV.AMB_WD][si]

                show_rotor_dict = {
                    "color": rotor_color,
                    "D": self.fres[FV.D][si],
                    "H": self.fres[FV.H][si],
                    "X": self.fres[FV.X][si],
                    "Y": self.fres[FV.Y][si],
                    "AMB_WD": self.fres[FV.AMB_WD][si],
                    "turb_angle": turb_angle,
                }
            else:
                show_rotor_dict = None

            out = get_fig(
                var=var,
                data=data[..., vi],
                si=si,
                s=s,
                x_pos=y_pos,
                y_pos=z_pos,
                title=ttl,
                add_bar=add_bar,
                vmin=vmin,
                vmax=vmax,
                quiv=quiv,
                invert_axis="x",
                animated=animated,
                show_rotor_dict=show_rotor_dict,
                rotor_plane="yz",
                rotor_slice={"axis": "x", "value": x_pos, "tol": 0.0},
                **kwargs,
            )

            yield out

    def precalc_chunk_xy(
        self,
        var: str,
        mdata: MData,
        fdata: FData,
        resolution: float = 100.0,
        figsize: tuple[int, int] = (8, 8),
        n_img_points: tuple[int, int] | None = None,
        xmin: float | None = None,
        ymin: float | None = None,
        xmax: float | None = None,
        ymax: float | None = None,
        z: float | None = None,
        xspace: float = 500.0,
        yspace: float = 500.0,
    ) -> tuple[Any, Any, Any]:
        """
        Pre-calculation of data for xy flow plots.

        Parameters
        ----------
        var
            The variable name
        mdata
            The model data
        fdata
            The farm data
        resolution
            The resolution in m
        figsize
            The figsize for plt.Figure
        n_img_points
            The number of image points along each axis
        xmin
            The minimal x position
        ymin
            The minimal y position
        xmax
            The maximal x position
        ymax
            The maximal y position
        z
            The z position
        xspace
            Additional space around turbines
        yspace
            Additional space around turbines

        Returns
        -------
        data
            The calculated data
        sinds
            The state indices
        gdata
            The grid data

        """
        gdata = get_grid_xy(
            self.fres,
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

        mlist, mpars = self.algo._collect_point_models()
        mlist.initialize(self.algo, verbosity=0, force=True)
        htdata = TData.from_points(gdata[-1], mdata=mdata)

        sinds = mdata[FC.STATE]
        data = mlist.calculate(self.algo, mdata, fdata, htdata, **mpars[0])
        data.pop(FV.WEIGHT, None)
        data = np2np_sp(data, sinds, gdata[0], gdata[1])

        return data, sinds, gdata
