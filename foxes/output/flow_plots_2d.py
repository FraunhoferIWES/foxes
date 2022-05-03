import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import foxes.variables as FV
import foxes.constants as FC
from foxes.output.output import Output


class FlowPlots2D(Output):
    """
    Class for horizontal or vertical 2D flow plots

    Parameters
    ----------
    algo: foxes.Algorithm
        The algorithm for point calculation
    farm_results: xarray.Dataset
        The farm results

    Attributes
    ----------
    algo: foxes.Algorithm
        The algorithm for point calculation
    farm_results: xarray.Dataset
        The farm results

    """

    def __init__(self, algo, farm_results):
        self.algo = algo
        self.fres = farm_results

    def _get_fig_hor(self, var, fig, figsize, ax, data, si, s, N_x, N_y, normalize_var, levels, 
                        x_pos, y_pos, vmin, vmax, cmap, xlabel, ylabel, title, add_bar, vlabel,
                        ret_state, ret_im):

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
                im = hax.pcolormesh(x_pos, y_pos,zz,vmin=vmin,vmax=vmax, shading='auto', cmap=cmap)
            
            # contour plot:
            else:
                im = hax.contourf(x_pos, y_pos,zz, levels, vmax=vmax, vmin=vmin, cmap=cmap)

            hax.autoscale_view()
            hax.set_xlabel(xlabel)
            hax.set_ylabel(ylabel)
            hax.set_title(title if title is not None else f"State {s}")
            plt.gca().set_aspect('equal', adjustable='box')

            if add_bar:
                divider = make_axes_locatable(hax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                vlab = vlabel if vlabel is not None else var
                hfig.colorbar(im, cax=cax, orientation='vertical', label=vlab)
                out = [hfig]
            else:
                out = [fig]

            if ret_state:
                out.append(si)
            if ret_im:
                out.append(im)
            
            return tuple(out)

    def get_mean_fig_horizontal(
            self,
            var,
            resolution, 
            xmin=None, ymin=None, 
            xmax=None, ymax=None, 
            xlabel='x [m]', ylabel='y [m]',
            z=None,
            xspace=500., yspace=500., 
            levels=None,var_min=None, var_max=None,
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
            **kwargs
        ):
        """
        Generates 2D farm flow figure in a plane.

        The kwargs are forwarded to the algorithm's calc_points
        function.

        Parameters
        ----------
        var: str
            The variable name
        resolution: float
            The resolution in m
        xaxis: np.array
            The x axis direction (normalized)
        zaxis: np.array
            The z axis direction (normalized)
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
        output_level: int
            The output level: 0 = silent, 1 = normal
        free_mem: bool
            Free memory. Switch off if after this calculation
            another one will be executed for the same wind farm
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
        add_bar: bool, optional
            Add a color bar
        cmap: str, optional
            The colormap
        weight_turbine: int, optional
            Index of the turbine from which to take the weight
        verbosity: int, optional
            The verbosity level
        ret_state: bool, optional
            Flag for state index return
        ret_im: bool, optional
            Flag for image return
        
        Yields
        ------
        fig: matplotlib.Figure
            The figure object
        si: int, optional
            The state index
        im: matplotlib.collections.QuadMesh or matplotlib.QuadContourSet, optional
            The image
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

        # find wind farm boundaries:
        """
        if len(self.farm.boundary_geometry):
            p_min = None
            p_max = None
            for b in self.farm.boundary_geometry:
                if p_min is None:
                    p_min = b.p_min()
                    p_max = b.p_max()
                else:
                    p_min = np.minimum(p_min, b.p_min()) 
                    p_max = np.maximum(p_max, b.p_max())
            bounds        = np.zeros([8, 3], dtype=FC.DTYPE)
            bounds[:4, 2] = np.min(rcentres[:, 2])
            bounds[4:, 2] = np.max(rcentres[:, 2])
            bounds[0, :2] = p_min[:2]
            bounds[1, :2] = np.array([p_min[0], p_max[1]])
            bounds[2, :2] = p_max[:2]
            bounds[3, :2] = np.array([p_max[0], p_min[1]])
            bounds[4:,:2] = bounds[:4, :2]
        else:
            bounds = rcentres
        """
        x_pos, x_res = np.linspace(x_min, x_max, num=int( (x_max - x_min) / resolution ) + 1, endpoint=True, retstep=True, dtype=None)
        y_pos, y_res = np.linspace(y_min, y_max, num=int( (y_max - y_min) / resolution ) + 1, endpoint=True, retstep=True, dtype=None)
        N_x, N_y     = len(x_pos), len(y_pos)
        n_pts        = len(x_pos) * len(y_pos)
        z_pos        = 0.5 * ( z_min + z_max )
        g_pts        = np.zeros((n_states, N_x, N_y, 3), dtype=FC.DTYPE)
        g_pts[:, :, :, 0] = x_pos[None, :, None]
        g_pts[:, :, :, 1] = y_pos[None, None, :]
        g_pts[:, :, :, 2] = z_pos
        g_pts = g_pts.reshape(n_states, n_pts, 3)

        if verbosity > 0:
            print("\nFlowPlots2D plot grid:")
            print("Min XYZ  =",x_min,y_min,z_min)
            print("Max XYZ  =",x_max,y_max,z_max)
            print("Pos Z    =",z_pos)
            print("Res XY   =",x_res,y_res)
            print("Dim XY   =",N_x,N_y)
            print("Grid pts =",n_pts)

        # calculate point results:
        point_results = self.algo.calc_points(
                            self.fres,
                            points=g_pts,
                            vars=[var],
                            **kwargs
                        )
        data = point_results[var].to_numpy()
        del point_results
        

        # take mean over states:
        weights = self.fres[FV.WEIGHT][:, weight_turbine].to_numpy()
        data    = np.einsum('s,sp->p', weights, data)

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
        out = self._get_fig_hor(var, fig, figsize, ax, data, None, None, N_x, N_y, 
                    normalize_var, levels, x_pos, y_pos, vmin, vmax, cmap, xlabel, ylabel, title, 
                    add_bar, vlabel, ret_state, ret_im)

        return out

    def gen_states_fig_horizontal(
            self,
            var,
            resolution, 
            xmin=None, ymin=None, 
            xmax=None, ymax=None, 
            xlabel='x [m]', ylabel='y [m]',
            z=None,
            xspace=500., yspace=500., 
            levels=None,var_min=None, var_max=None,
            figsize=None,
            normalize_xy=None,
            normalize_var=None,
            title=None,
            vlabel=None,
            fig=None,
            ax=None,
            add_bar=True,
            cmap=None,
            verbosity=1,
            ret_state=False,
            ret_im=False,
            **kwargs
        ):
        """
        Generates 2D farm flow figure in a plane.

        The kwargs are forwarded to the algorithm's calc_points
        function.

        Parameters
        ----------
        var: str
            The variable name
        resolution: float
            The resolution in m
        xaxis: np.array
            The x axis direction (normalized)
        zaxis: np.array
            The z axis direction (normalized)
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
        output_level: int
            The output level: 0 = silent, 1 = normal
        free_mem: bool
            Free memory. Switch off if after this calculation
            another one will be executed for the same wind farm
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
        add_bar: bool, optional
            Add a color bar
        cmap: str, optional
            The colormap
        verbosity: int, optional
            The verbosity level
        ret_state: bool, optional
            Flag for state index return
        ret_im: bool, optional
            Flag for image return
        
        Yields
        ------
        fig: matplotlib.Figure
            The figure object
        si: int, optional
            The state index
        im: matplotlib.collections.QuadMesh or matplotlib.QuadContourSet, optional
            The image
        
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

        # find wind farm boundaries:
        """
        if len(self.farm.boundary_geometry):
            p_min = None
            p_max = None
            for b in self.farm.boundary_geometry:
                if p_min is None:
                    p_min = b.p_min()
                    p_max = b.p_max()
                else:
                    p_min = np.minimum(p_min, b.p_min()) 
                    p_max = np.maximum(p_max, b.p_max())
            bounds        = np.zeros([8, 3], dtype=FC.DTYPE)
            bounds[:4, 2] = np.min(rcentres[:, 2])
            bounds[4:, 2] = np.max(rcentres[:, 2])
            bounds[0, :2] = p_min[:2]
            bounds[1, :2] = np.array([p_min[0], p_max[1]])
            bounds[2, :2] = p_max[:2]
            bounds[3, :2] = np.array([p_max[0], p_min[1]])
            bounds[4:,:2] = bounds[:4, :2]
        else:
            bounds = rcentres
        """
        x_pos, x_res = np.linspace(x_min, x_max, num=int( (x_max - x_min) / resolution ) + 1, endpoint=True, retstep=True, dtype=None)
        y_pos, y_res = np.linspace(y_min, y_max, num=int( (y_max - y_min) / resolution ) + 1, endpoint=True, retstep=True, dtype=None)
        N_x, N_y     = len(x_pos), len(y_pos)
        n_pts        = len(x_pos) * len(y_pos)
        z_pos        = 0.5 * ( z_min + z_max )
        g_pts        = np.zeros((n_states, N_x, N_y, 3), dtype=FC.DTYPE)
        g_pts[:, :, :, 0] = x_pos[None, :, None]
        g_pts[:, :, :, 1] = y_pos[None, None, :]
        g_pts[:, :, :, 2] = z_pos
        g_pts = g_pts.reshape(n_states, n_pts, 3)

        if verbosity > 0:
            print("\nFlowPlots2D plot grid:")
            print("Min XYZ  =",x_min,y_min,z_min)
            print("Max XYZ  =",x_max,y_max,z_max)
            print("Pos Z    =",z_pos)
            print("Res XY   =",x_res,y_res)
            print("Dim XY   =",N_x,N_y)
            print("Grid pts =",n_pts)

        # calculate point results:
        point_results = self.algo.calc_points(
                            self.fres,
                            points=g_pts,
                            vars=[var],
                            **kwargs
                        )
        data = point_results[var].to_numpy()
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

            out = self._get_fig_hor(var, fig, figsize, ax, data, si, s, N_x, N_y, normalize_var,
                        levels, x_pos, y_pos, vmin, vmax, cmap, xlabel, ylabel, title, add_bar, 
                        vlabel, ret_state, ret_im)
            
            yield out
                 