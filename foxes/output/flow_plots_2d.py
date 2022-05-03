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

    def get_mean_flow_figure(
            self,
            model_book,
            var,
            resolution, 
            states_sel=None, 
            point_data_model=None,
            xaxis=np.array([1,0,0]),zaxis=np.array([0,0,1]), 
            xmin=None, ymin=None, 
            xmax=None, ymax=None, 
            xlabel='x [m]', ylabel='y [m]',
            z=None,
            xspace=500., yspace=500., 
            levels=None,var_min=None, var_max=None,
            output_level=1,
            free_mem=False,
            add_bar=True,
            figsize=None,
            zorder=None,
            cmap=None,
            fig=None,
            ax=None,
            normalize_xy=None,
            normalize_var=None,
            title=None,
            vlabel=None,
            **kwargs
        ):
        """
        Generates the mean 2D farm flow figure in a plane.

        The kwargs are forwarded to the calculation algorithm.

        Parameters
        ----------
        model_book: flappy.ModelBook
            The model book
        var: str
            The variable name
        resolution: float
            The resolution in m
        states_sel: list
            List of states to plot, or None for all
        point_data_model: flappy.PointDataModel
            The model in which var is calculated,
            or None for PDMBasicPointData.
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
        free_mem: bool
            Free memory. Switch off if after this calculation
            another one will be executed for the same wind farm
        add_bar: bool
            Add colorbar
        output_level: int
            The output level: 0 = silent, 1 = normal
        zorder: int
            The zorder for plotting
        cmap: str
            The colormap
        fig: matplotlib.pyplot.Figure, optional
            The figure object to which to add
        ax: matplotlib.pyplot.Axis, optional
            The axis object, to which to add
        normalize_xy: float, optional
            Divide x and y by this value
        normalize_var: float, optional
            Divide the variable by this value
        title: str, optional
            The title
        vlabel: str, optional
            The variable label

        Returns
        -------
        ax: matplotlib.pyplot.Axis
            The axis object
        im: matplotlib.collections.QuadMesh or matplotlib.QuadContourSet
            The image
        
        """
        # prepare:
        ssel = self.amb_states.states_sel if states_sel is None \
                    else IndexSelection(self.amb_states.states_sel, selection=states_sel)
        n_states = ssel.size

        # get turbine coordinates:
        rcentres = self.farm_data.turbine_results[[FV.X, FV.Y, FV.H]].values

        # find wind farm boundaries:
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

        yaxis        = np.cross(zaxis,xaxis)
        x_min        = xmin if xmin is not None else np.min(xaxis.dot(bounds.T)) - xspace
        y_min        = ymin if ymin is not None else np.min(yaxis.dot(bounds.T)) - yspace
        z_min        = z if z is not None else np.min(zaxis.dot(bounds.T)) 
        x_max        = xmax if xmax is not None else np.max(xaxis.dot(bounds.T)) + xspace
        y_max        = ymax if ymax is not None else np.max(yaxis.dot(bounds.T)) + yspace
        z_max        = z if z is not None else np.max(zaxis.dot(bounds.T))
        x_pos, x_res = np.linspace(x_min, x_max, num=int( (x_max - x_min) / resolution ) + 1, endpoint=True, retstep=True, dtype=None)
        y_pos, y_res = np.linspace(y_min, y_max, num=int( (y_max - y_min) / resolution ) + 1, endpoint=True, retstep=True, dtype=None)
        N_x, N_y     = len(x_pos), len(y_pos)
        n_pts        = len(x_pos) * len(y_pos)
        z_pos        = 0.5 * ( z_min + z_max )
        g_pts        = np.zeros([n_pts, 3])
        counter = 0
        for xi in range(N_x):
            for yi in range(N_y):
                g_pts[counter] += x_pos[xi] * xaxis
                g_pts[counter] += y_pos[yi] * yaxis
                g_pts[counter] += z_pos     * zaxis
                counter        += 1

        if output_level:
            print("\nFlowPlot2DOutput plot grid:")
            print("Min XYZ  =",x_min,y_min,z_min)
            print("Max XYZ  =",x_max,y_max,z_max)
            print("Pos Z    =",z_pos)
            print("Res XY   =",x_res,y_res)
            print("Dim XY   =",N_x,N_y)
            print("Grid pts =",counter)

        # create flow states:
        flow_states = FarmFlowStates(self.amb_states, self.farm, self.farm_data)

        # calculate point results:
        point_results = flow_states.get_wind_ti_rho(
                            model_book,
                            g_pts,
                            states_sel=ssel,
                            output_level=output_level,
                            free_mem=free_mem,
                            **kwargs
                        ).values
        
        # calculate point data:
        data_model = point_data_model if point_data_model is not None \
                        else PDMBasicPointData()
        data = data_model.calc_vars(point_results.reshape([n_states * n_pts, 10]), 
                            vars=[var]).values[:, 0].reshape([n_states, n_pts])

        # take mean over states:
        weights = self.farm_data.state_results[FV.WEIGHT].values
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

        if fig is None:
            fig = plt.figure(figsize=figsize)
            ax  = fig.add_subplot(111)
        else:
            ax = fig.axes[0] if ax is None else ax

        # get result data:
        zz = data.reshape([N_x,N_y]).T
        if normalize_var is not None:
            zz /= normalize_var

        # raw data image:
        if levels is None:
            im = ax.pcolormesh(x_pos, y_pos,zz,vmin=vmin,vmax=vmax, 
                    cmap=cmap, shading='auto', zorder=zorder)
        
        # contour plot:
        else:
            im = ax.contourf(x_pos, y_pos,zz, levels, vmax=vmax, vmin=vmin,
                                cmap=cmap, zorder=zorder)

        ax.autoscale_view()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        plt.gca().set_aspect('equal', adjustable='box')

        if add_bar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            vlab = vlabel if vlabel is not None else var
            fig.colorbar(im, cax=cax, orientation='vertical', label=vlab)

        return ax, im

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
            verbosity=1,
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
        verbosity: int, optional
            The verbosity level
        ret_im: bool, optional
            Flag for return image
        
        Yields
        ------
        si: int
            The state index
        fig: matplotlib.Figure
            The figure object
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
        
        bounds = rcentres
        
        yaxis        = np.cross(zaxis,xaxis)
        x_min        = xmin if xmin is not None else np.min(xaxis.dot(bounds.T)) - xspace
        y_min        = ymin if ymin is not None else np.min(yaxis.dot(bounds.T)) - yspace
        z_min        = z if z is not None else np.min(zaxis.dot(bounds.T)) 
        x_max        = xmax if xmax is not None else np.max(xaxis.dot(bounds.T)) + xspace
        y_max        = ymax if ymax is not None else np.max(yaxis.dot(bounds.T)) + yspace
        z_max        = z if z is not None else np.max(zaxis.dot(bounds.T))
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
            print("\nFlowPlot2DOutput plot grid:")
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
            zz = data[si].reshape([N_x,N_y]).T
            if normalize_var is not None:
                zz /= normalize_var

            # raw data image:
            if levels is None:
                im = hax.pcolormesh(x_pos, y_pos,zz,vmin=vmin,vmax=vmax, shading='auto')
            
            # contour plot:
            else:
                im = hax.contourf(x_pos, y_pos,zz, levels, vmax=vmax, vmin=vmin)

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

                ofig = hfig
            
            else:
                ofig = fig
            
            if ret_im:
                yield s, ofig, im
            
            else:
                yield s, ofig
                 