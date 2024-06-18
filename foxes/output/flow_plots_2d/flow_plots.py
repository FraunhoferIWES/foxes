import numpy as np

from foxes.output import SliceData
import foxes.variables as FV

from .get_fig import get_fig


class FlowPlots2D(SliceData):
    """
    Class for horizontal or vertical 2D flow plots

    :group: output.flow_plots_2d

    """

    def get_mean_fig_xy(
        self,
        var,
        xlabel="x [m]",
        ylabel="y [m]",
        levels=None,
        figsize=None,
        title=None,
        vlabel=None,
        fig=None,
        ax=None,
        add_bar=True,
        cmap=None,
        vmin=None,
        vmax=None,
        quiver_n=None,
        quiver_pars={},
        ret_state=False,
        ret_im=False,
        ret_data=False,
        animated=False,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a horizontal xy-plane.

        Parameters
        ----------
        var: str
            The variable name
        xlabel: str, optional
            The x axis label
        ylabel: str, optional
            The y axis label
        levels: int, optional
            The number of levels for the contourf plot, or None for pure image
        figsize: tuple, optional
            The figsize for plt.Figure
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
        vmin: float, optional
            The minimal variable value
        vmax: float, optional
            The maximal variable value
        quiver_n: int, optional
            Place a vector at each `n`th point
        quiver_pars: dict, optional
            Parameters for plt.quiver
        ret_state: bool
            Flag for state index return
        ret_im: bool
            Flag for image return
        ret_data: bool
            Flag for returning image data
        animated: bool
            Switch for usage for an animation
        kwargs: dict, optional
            Additional parameters for SliceData.get_mean_data_xy

        Returns
        -------
        fig: matplotlib.Figure
            The figure object
        si: int, optional
            The state index
        im: tuple, optional
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet
        data: numpy.ndarray, optional
            The image data, shape: (n_x, n_y)

        """
        variables = list(set([var] + [FV.WD, FV.WS]))
        vi = variables.index(var)
        wdi = variables.index(FV.WD)
        wsi = variables.index(FV.WS)

        data, gdata = self.get_mean_data_xy(
            variables=variables,
            vmin={var: vmin} if vmin is not None else {},
            vmax={var: vmax} if vmax is not None else {},
            data_format="numpy",
            ret_grid=True,
            **kwargs,
        )
        x_pos, y_pos, z_pos, __ = gdata

        if title is None:
            title = f"States mean, z =  {int(np.round(z_pos))} m"

        # define wind vector arrows:
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
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
        var,
        x_direction=270,
        xlabel="x [m]",
        zlabel="z [m]",
        levels=None,
        figsize=None,
        title=None,
        vlabel=None,
        fig=None,
        ax=None,
        add_bar=True,
        cmap=None,
        vmin=None,
        vmax=None,
        quiver_n=None,
        quiver_pars={},
        ret_state=False,
        ret_im=False,
        ret_data=False,
        animated=False,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a horizontal xz-plane.

        Parameters
        ----------
        var: str
            The variable name
        x_direction: float, optional
            The direction of the x axis, 0 = north
        xlabel: str, optional
            The x axis label
        zlabel: str, optional
            The z axis label
        levels: int, optional
            The number of levels for the contourf plot, or None for pure image
        figsize: tuple, optional
            The figsize for plt.Figure
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
        vmin: float, optional
            The minimal variable value
        vmax: float, optional
            The maximal variable value
        quiver_n: int, optional
            Place a vector at each `n`th point
        quiver_pars: dict, optional
            Parameters for plt.quiver
        ret_state: bool
            Flag for state index return
        ret_im: bool
            Flag for image return
        ret_data: bool
            Flag for returning image data
        animated: bool
            Switch for usage for an animation
        kwargs: dict, optional
            Additional parameters for SliceData.get_mean_data_xz

        Returns
        -------
        fig: matplotlib.Figure
            The figure object
        si: int, optional
            The state index
        im: tuple, optional
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet
        data: numpy.ndarray, optional
            The image data, shape: (n_x, n_y)

        """
        variables = list(set([var] + [FV.WD, FV.WS]))
        vi = variables.index(var)
        wdi = variables.index(FV.WD)
        wsi = variables.index(FV.WS)

        data, gdata = self.get_mean_data_xz(
            variables=variables,
            vmin={var: vmin} if vmin is not None else {},
            vmax={var: vmax} if vmax is not None else {},
            x_direction=x_direction,
            data_format="numpy",
            ret_grid=True,
            **kwargs,
        )
        x_pos, y_pos, z_pos, __ = gdata

        if title is None:
            title = f"States mean, x direction {x_direction}°, y =  {int(np.round(y_pos))} m"

        # define wind vector arrows:
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
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
        var,
        x_direction=270,
        ylabel="x [m]",
        zlabel="z [m]",
        levels=None,
        figsize=None,
        title=None,
        vlabel=None,
        fig=None,
        ax=None,
        add_bar=True,
        cmap=None,
        vmin=None,
        vmax=None,
        quiver_n=None,
        quiver_pars={},
        ret_state=False,
        ret_im=False,
        ret_data=False,
        animated=False,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a horizontal yz-plane.

        Parameters
        ----------
        var: str
            The variable name
        x_direction: float, optional
            The direction of the x axis, 0 = north
        ylabel: str, optional
            The y axis label
        zlabel: str, optional
            The z axis label
        levels: int, optional
            The number of levels for the contourf plot, or None for pure image
        figsize: tuple, optional
            The figsize for plt.Figure
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
        vmin: float, optional
            The minimal variable value
        vmax: float, optional
            The maximal variable value
        quiver_n: int, optional
            Place a vector at each `n`th point
        quiver_pars: dict, optional
            Parameters for plt.quiver
        ret_state: bool
            Flag for state index return
        ret_im: bool
            Flag for image return
        ret_data: bool
            Flag for returning image data
        animated: bool
            Switch for usage for an animation
        kwargs: dict, optional
            Additional parameters for SliceData.get_mean_data_yz

        Returns
        -------
        fig: matplotlib.Figure
            The figure object
        si: int, optional
            The state index
        im: tuple, optional
            The image objects, matplotlib.collections.QuadMesh
            or matplotlib.QuadContourSet
        data: numpy.ndarray, optional
            The image data, shape: (n_x, n_y)

        """
        variables = list(set([var] + [FV.WD, FV.WS]))
        vi = variables.index(var)
        wdi = variables.index(FV.WD)
        wsi = variables.index(FV.WS)

        data, gdata = self.get_mean_data_yz(
            variables=variables,
            vmin={var: vmin} if vmin is not None else {},
            vmax={var: vmax} if vmax is not None else {},
            x_direction=x_direction,
            data_format="numpy",
            ret_grid=True,
            **kwargs,
        )
        x_pos, y_pos, z_pos, __ = gdata

        if title is None:
            title = f"States mean, x direction {x_direction}°, x =  {int(np.round(x_pos))} m"

        # define wind vector arrows:
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
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

    def gen_states_fig_xy(
        self,
        var,
        xlabel="x [m]",
        ylabel="y [m]",
        levels=None,
        figsize=None,
        title=None,
        vlabel=None,
        fig=None,
        ax=None,
        add_bar=True,
        cmap=None,
        vmin=None,
        vmax=None,
        quiver_n=None,
        quiver_pars={},
        ret_state=False,
        ret_im=False,
        animated=False,
        rotor_color=None,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a horizontal xy-plane.

        Parameters
        ----------
        var: str
            The variable name
        xlabel: str, optional
            The x axis label
        ylabel: str, optional
            The y axis label
        levels: int, optional
            The number of levels for the contourf plot, or None for pure image
        figsize: tuple, optional
            The figsize for plt.Figure
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
        vmin: float, optional
            The minimal variable value
        vmax: float, optional
            The maximal variable value
        quiver_n: int, optional
            Place a vector at each `n`th point
        quiver_pars: dict, optional
            Parameters for plt.quiver
        ret_state: bool
            Flag for state index return
        ret_im: bool
            Flag for image return
        animated: bool
            Switch for usage for an animation
        rotor_color: str, optional
            Indicate the rotor orientation by a colored line
        kwargs: dict, optional
            Additional parameters for SliceData.get_states_data_xy

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
        variables = list(set([var] + [FV.WD, FV.WS]))
        vi = variables.index(var)
        wdi = variables.index(FV.WD)
        wsi = variables.index(FV.WS)

        data, states, gdata = self.get_states_data_xy(
            variables=variables,
            vmin={var: vmin} if vmin is not None else {},
            vmax={var: vmax} if vmax is not None else {},
            data_format="numpy",
            ret_states=True,
            ret_grid=True,
            **kwargs,
        )
        x_pos, y_pos, z_pos, __ = gdata

        # define wind vector arrows:
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
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
                ttl += f", z =  {int(np.round(z_pos))} m"
            elif callable(title):
                ttl = title(si, s)
            else:
                ttl = title

            # get data for show_turbines
            if rotor_color is not None:
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
                fig=fig,
                figsize=figsize,
                ax=ax,
                data=data[..., vi],
                si=si,
                s=s,
                levels=levels,
                x_pos=x_pos,
                y_pos=y_pos,
                cmap=cmap,
                xlabel=xlabel,
                ylabel=ylabel,
                title=ttl,
                add_bar=add_bar,
                vlabel=vlabel,
                vmin=vmin,
                vmax=vmax,
                quiv=quiv,
                ret_state=ret_state,
                ret_im=ret_im,
                animated=animated,
                show_rotor_dict=show_rotor_dict,
            )

            yield out

    def gen_states_fig_xz(
        self,
        var,
        x_direction=270.0,
        xlabel="x [m]",
        zlabel="z [m]",
        levels=None,
        figsize=None,
        title=None,
        vlabel=None,
        fig=None,
        ax=None,
        add_bar=True,
        cmap=None,
        vmin=None,
        vmax=None,
        quiver_n=None,
        quiver_pars={},
        ret_state=False,
        ret_im=False,
        animated=False,
        rotor_color=None,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a vertical xz-plane.

        Parameters
        ----------
        var: str
            The variable name
        x_direction: float, optional
            The direction of the x axis, 0 = north
        xlabel: str, optional
            The x axis label
        zlabel: str, optional
            The z axis label
        levels: int, optional
            The number of levels for the contourf plot, or None for pure image
        figsize: tuple, optional
            The figsize for plt.Figure
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
        vmin: float, optional
            The minimal variable value
        vmax: float, optional
            The maximal variable value
        quiver_n: int, optional
            Place a vector at ech `n`th point
        quiver_pars: dict, optional
            Parameters for plt.quiver
        ret_state: bool
            Flag for state index return
        ret_im: bool
            Flag for image return
        animated: bool
            Switch for usage for an animation
        rotor_color: str, optional
            Indicate the rotor orientation by a colored line
        kwargs: dict, optional
            Additional parameters for SliceData.get_states_data_xz

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
        variables = list(set([var] + [FV.WD, FV.WS]))
        vi = variables.index(var)
        wdi = variables.index(FV.WD)
        wsi = variables.index(FV.WS)

        data, states, gdata = self.get_states_data_xz(
            variables=variables,
            vmin={var: vmin} if vmin is not None else {},
            vmax={var: vmax} if vmax is not None else {},
            data_format="numpy",
            ret_states=True,
            ret_grid=True,
            x_direction=x_direction,
            **kwargs,
        )
        x_pos, y_pos, z_pos, __ = gdata

        # define wind vector arrows:
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
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
                fig=fig,
                figsize=figsize,
                ax=ax,
                data=data[..., vi],
                si=si,
                s=s,
                levels=levels,
                x_pos=x_pos,
                y_pos=z_pos,
                cmap=cmap,
                xlabel=xlabel,
                ylabel=zlabel,
                title=ttl,
                add_bar=add_bar,
                vlabel=vlabel,
                quiv=quiv,
                vmin=vmin,
                vmax=vmax,
                ret_state=ret_state,
                ret_im=ret_im,
                animated=animated,
                show_rotor_dict=show_rotor_dict,
            )

            yield out

    def gen_states_fig_yz(
        self,
        var,
        x_direction=270.0,
        ylabel="y [m]",
        zlabel="z [m]",
        levels=None,
        figsize=None,
        title=None,
        vlabel=None,
        fig=None,
        ax=None,
        add_bar=True,
        cmap=None,
        vmin=None,
        vmax=None,
        quiver_n=None,
        quiver_pars={},
        ret_state=False,
        ret_im=False,
        animated=False,
        rotor_color=None,
        **kwargs,
    ):
        """
        Generates 2D farm flow figure in a vertical yz-plane.

        Parameters
        ----------
        var: str
            The variable name
        x_direction: float, optional
            The direction of the x axis, 0 = north
        ylabel: str, optional
            The y axis label
        zlabel: str, optional
            The z axis label
        levels: int, optional
            The number of levels for the contourf plot, or None for pure image
        figsize: tuple, optional
            The figsize for plt.Figure
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
        vmin: float, optional
            The minimal variable value
        vmax: float, optional
            The maximal variable value
        quiver_n: int, optional
            Place a vector at ech `n`th point
        quiver_pars: dict, optional
            Parameters for plt.quiver
        ret_state: bool
            Flag for state index return
        ret_im: bool
            Flag for image return
        animated: bool
            Switch for usage for an animation
        rotor_color: str, optional
            Indicate the rotor orientation by a colored line
        kwargs: dict, optional
            Additional parameters for SliceData.get_states_data_yz

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
        variables = list(set([var] + [FV.WD, FV.WS]))
        vi = variables.index(var)
        wdi = variables.index(FV.WD)
        wsi = variables.index(FV.WS)

        data, states, gdata = self.get_states_data_yz(
            variables=variables,
            vmin={var: vmin} if vmin is not None else {},
            vmax={var: vmax} if vmax is not None else {},
            data_format="numpy",
            ret_states=True,
            ret_grid=True,
            x_direction=x_direction,
            **kwargs,
        )
        x_pos, y_pos, z_pos, __ = gdata

        # define wind vector arrows:
        qpars = dict(angles="xy", scale_units="xy", scale=0.05)
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
                fig=fig,
                figsize=figsize,
                ax=ax,
                data=data[..., vi],
                si=si,
                s=s,
                levels=levels,
                x_pos=y_pos,
                y_pos=z_pos,
                cmap=cmap,
                xlabel=ylabel,
                ylabel=zlabel,
                title=ttl,
                add_bar=add_bar,
                vlabel=vlabel,
                vmin=vmin,
                vmax=vmax,
                quiv=quiv,
                ret_state=ret_state,
                ret_im=ret_im,
                invert_axis="x",
                animated=animated,
                show_rotor_dict=show_rotor_dict,
            )

            yield out
