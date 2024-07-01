import numpy as np

from foxes.output import Output
from foxes.utils import write_nc
import foxes.constants as FC
import foxes.variables as FV

from . import grids


class SliceData(Output):
    """
    Create data for horizontal or vertical 2D slices

    Attributes
    ----------
    algo: foxes.Algorithm
        The algorithm for point calculation
    farm_results: xarray.Dataset
        The farm results
    runner: foxes.utils.runners.Runner, optional
        The runner
    verbosity_delta: int
        Verbosity threshold for printing calculation info
            
    :group: output

    """

    def __init__(
        self, 
        algo, 
        farm_results, 
        runner=None, 
        verbosity_delta=1,
        **kwargs,
        ):
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
        verbosity_delta: int
            Verbosity threshold for printing calculation info
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(**kwargs)
        self.algo = algo
        self.fres = farm_results
        self.runner = runner
        self.verbosity_delta = verbosity_delta

    def _data_mod(
        self,
        a_pos,
        b_pos,
        c_pos,
        data,
        normalize_a,
        normalize_b,
        normalize_c,
        normalize_v,
        vmin,
        vmax,
    ):
        """Helper function for data modification"""
        if normalize_a is not None:
            a_pos /= normalize_a
        if normalize_b is not None:
            b_pos /= normalize_b
        if normalize_c is not None:
            c_pos /= normalize_c

        for v in data:
            if v in normalize_v:
                data[v] /= normalize_v[v]
            if v in vmin:
                data[v] = np.maximum(data[v], vmin[v])
            if v in vmax:
                data[v] = np.minimum(data[v], vmax[v])

        return a_pos, b_pos, c_pos, data

    def _write(self, format, data, fname, verbosity, **write_pars):
        """Helper function for file writing"""
        if fname is not None:
            if format == "numpy":
                fpath = self.get_fpath(fname)
                if verbosity > 0:
                    print("Writing file", fpath)
                wpars = dict(format="%.6f")
                wpars.update(write_pars)
                data.tofile(fpath, **wpars)

            elif format == "pandas":
                if verbosity > 0:
                    print("Writing file", fpath)
                self.write(fname, data, **write_pars)

            elif format == "xarray":
                write_nc(data, self.get_fpath(fname), verbosity=verbosity, **write_pars)

            else:
                raise ValueError(
                    f"Unknown data format '{format}', choices: numpy, pandas, xarray"
                )

    def _calc_mean_data(
        self,
        ori,
        data_format,
        variables,
        a_pos,
        b_pos,
        c_pos,
        g_pts,
        normalize_a,
        normalize_b,
        normalize_c,
        normalize_v,
        label_map,
        vmin,
        vmax,
        states_sel,
        states_isel,
        weight_turbine,
        to_file,
        write_pars,
        ret_states,
        verbosity,
        **kwargs,
    ):
        """Helper function for mean data calculation"""
        # calculate point results:
        point_results = grids.calc_point_results(
            algo=self.algo,
            farm_results=self.fres,
            g_pts=g_pts,
            sel={FC.STATE: states_sel} if states_sel is not None else None,
            isel={FC.STATE: states_isel} if states_isel is not None else None,
            verbosity=verbosity-self.verbosity_delta,
            **kwargs,
        )
        states = point_results[FC.STATE].to_numpy()
        if variables is None:
            variables = list(point_results.data_vars.keys())
        else:
            point_results.drop_vars(variables)
        del g_pts

        # take mean over states:
        weights = self.fres[FV.WEIGHT][:, weight_turbine].to_numpy()
        data = {
            v: np.einsum("s,sp->p", weights, point_results[v].to_numpy())
            for v in variables
        }
        del point_results

        # apply data modification:
        a_pos, b_pos, c_pos, data = self._data_mod(
            a_pos,
            b_pos,
            c_pos,
            data,
            normalize_a,
            normalize_b,
            normalize_c,
            normalize_v,
            vmin,
            vmax,
        )

        # translate to selected format:
        if data_format == "numpy":
            data = grids.np2np_p(data, a_pos, b_pos)
            self._write(data_format, data, to_file, verbosity, **write_pars)
        elif data_format == "pandas":
            data = grids.np2pd_p(data, a_pos, b_pos, ori, label_map)
            self._write(data_format, data, to_file, verbosity, **write_pars)
        elif data_format == "xarray":
            data = grids.np2xr_p(data, a_pos, b_pos, c_pos, ori, label_map)
            self._write(data_format, data, to_file, verbosity, **write_pars)
        else:
            raise ValueError(
                f"Unknown data format '{data_format}', choices: numpy, pandas, xarray"
            )

        return (data, states) if ret_states else data

    def get_mean_data_xy(
        self,
        resolution,
        variables=None,
        data_format="xarray",
        xmin=None,
        ymin=None,
        xmax=None,
        ymax=None,
        z=None,
        xspace=500.0,
        yspace=500.0,
        normalize_x=None,
        normalize_y=None,
        normalize_z=None,
        normalize_v={},
        label_map={},
        vmin={},
        vmax={},
        states_sel=None,
        states_isel=None,
        weight_turbine=0,
        to_file=None,
        write_pars={},
        ret_states=False,
        ret_grid=False,
        verbosity=0,
        **kwargs,
    ):
        """
        Creates mean data of 2D farm flow slices in a horizontal xy-plane.

        Parameters
        ----------
        resolution: float
            The resolution in m
        variables: list of str, optional
            The variables, or None for all
        data_format: str
            The output data format: numpy, pandas, xarray
        xmin: float, optional
            The min x coordinate, or None for automatic
        ymin: float, optional
            The min y coordinate, or None for automatic
        xmax: float, optional
            The max x coordinate, or None for automatic
        ymax: float, optional
            The max y coordinate, or None for automatic
        z: float, optional
            The z coordinate of the plane
        xspace: float, optional
            The extra space in x direction, before and after wind farm
        yspace: float, optional
            The extra space in y direction, before and after wind farm
        normalize_x: float, optional
            Divide x by this value
        normalize_y: float, optional
            Divide y by this value
        normalize_z: float, optional
            Divide z by this value
        normalize_v: dict, optional
            Divide the variables by these values
        label_map: dict
            The mapping from original to new field names
        vmin: dict
            Minimal values for variables
        vmax: dict
            Maximal values for variables
        states_sel: list, optional
            Reduce to selected states
        states_isel: list, optional
            Reduce to the selected states indices
        weight_turbine: int, optional
            Index of the turbine from which to take the weight
        to_file: str, optional
            Write data to this file name
        write_pars: dict
            Additional write function parameters
        ret_states: bool
            Flag for returning states indices
        ret_grid: bool
            Flag for returning grid data
        verbosity: int, optional
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Returns
        -------
        data: dict or pandas.DataFrame or xarray.Dataset
            The gridded data
        states: numpy.ndarray, optional
            The states indices
        grid_data: tuple, optional
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        gdata = grids.get_grid_xy(
            self.fres, 
            resolution, 
            xmin, 
            ymin, 
            xmax, 
            ymax, 
            z, 
            xspace, 
            yspace, 
            verbosity-self.verbosity_delta,
        )

        data = self._calc_mean_data(
            "xy",
            data_format,
            variables,
            *gdata,
            normalize_x,
            normalize_y,
            normalize_z,
            normalize_v,
            label_map,
            vmin,
            vmax,
            states_sel,
            states_isel,
            weight_turbine,
            to_file,
            write_pars,
            ret_states,
            verbosity,
            **kwargs,
        )

        if ret_grid:
            out = list(data) if ret_states else [data]
            return tuple(out + [gdata])
        return data

    def get_mean_data_xz(
        self,
        resolution,
        variables=None,
        data_format="xarray",
        x_direction=270,
        xmin=None,
        zmin=0.0,
        xmax=None,
        zmax=None,
        y=None,
        xspace=500.0,
        zspace=500.0,
        normalize_x=None,
        normalize_y=None,
        normalize_z=None,
        normalize_v={},
        label_map={},
        vmin={},
        vmax={},
        states_sel=None,
        states_isel=None,
        weight_turbine=0,
        to_file=None,
        write_pars={},
        ret_states=False,
        ret_grid=False,
        verbosity=0,
        **kwargs,
    ):
        """
        Creates mean data of 2D farm flow slices in an xz-plane.

        Parameters
        ----------
        resolution: float
            The resolution in m
        variables: list of str, optional
            The variables, or None for all
        data_format: str
            The output data format: numpy, pandas, xarray
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
        y: float, optional
            The y coordinate of the plane
        xspace: float, optional
            The extra space in x direction, before and after wind farm
        zspace: float, optional
            The extra space in z direction, below and above wind farm
        normalize_x: float, optional
            Divide x by this value
        normalize_y: float, optional
            Divide y by this value
        normalize_z: float, optional
            Divide z by this value
        normalize_v: dict, optional
            Divide the variables by these values
        label_map: dict
            The mapping from original to new field names
        vmin: dict
            Minimal values for variables
        vmax: dict
            Maximal values for variables
        states_sel: list, optional
            Reduce to selected states
        states_isel: list, optional
            Reduce to the selected states indices
        weight_turbine: int, optional
            Index of the turbine from which to take the weight
        to_file: str, optional
            Write data to this file name
        write_pars: dict
            Additional write function parameters
        ret_states: bool
            Flag for returning states indices
        ret_grid: bool
            Flag for returning grid data
        verbosity: int, optional
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Returns
        -------
        data: dict or pandas.DataFrame or xarray.Dataset
            The gridded data
        states: numpy.ndarray, optional
            The states indices
        grid_data: tuple, optional
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        gdata = grids.get_grid_xz(
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
            verbosity-self.verbosity_delta,
        )
        gdatb = (gdata[0], gdata[2], gdata[1], gdata[3])

        data = self._calc_mean_data(
            "xz",
            data_format,
            variables,
            *gdatb,
            normalize_x,
            normalize_z,
            normalize_y,
            normalize_v,
            label_map,
            vmin,
            vmax,
            states_sel,
            states_isel,
            weight_turbine,
            to_file,
            write_pars,
            ret_states,
            verbosity,
            **kwargs,
        )

        if ret_grid:
            out = list(data) if ret_states else [data]
            return tuple(out + [gdata])
        return data

    def get_mean_data_yz(
        self,
        resolution,
        variables=None,
        data_format="xarray",
        x_direction=270,
        ymin=None,
        zmin=0.0,
        ymax=None,
        zmax=None,
        x=None,
        yspace=500.0,
        zspace=500.0,
        normalize_x=None,
        normalize_y=None,
        normalize_z=None,
        normalize_v={},
        label_map={},
        vmin={},
        vmax={},
        states_sel=None,
        states_isel=None,
        weight_turbine=0,
        to_file=None,
        write_pars={},
        ret_states=False,
        ret_grid=False,
        verbosity=0,
        **kwargs,
    ):
        """
        Creates mean data of 2D farm flow slices in a yz-plane.

        Parameters
        ----------
        resolution: float
            The resolution in m
        variables: list of str, optional
            The variables, or None for all
        data_format: str
            The output data format: numpy, pandas, xarray
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
        x: float, optional
            The x coordinate of the plane
        yspace: float, optional
            The extra space in y direction, before and after wind farm
        zspace: float, optional
            The extra space in z direction, below and above wind farm
        normalize_x: float, optional
            Divide x by this value
        normalize_y: float, optional
            Divide y by this value
        normalize_z: float, optional
            Divide z by this value
        normalize_v: dict, optional
            Divide the variables by these values
        label_map: dict
            The mapping from original to new field names
        vmin: dict
            Minimal values for variables
        vmax: dict
            Maximal values for variables
        states_sel: list, optional
            Reduce to selected states
        states_isel: list, optional
            Reduce to the selected states indices
        weight_turbine: int, optional
            Index of the turbine from which to take the weight
        to_file: str, optional
            Write data to this file name
        write_pars: dict
            Additional write function parameters
        ret_states: bool
            Flag for returning states indices
        ret_grid: bool
            Flag for returning grid data
        verbosity: int, optional
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Returns
        -------
        data: dict or pandas.DataFrame or xarray.Dataset
            The gridded data
        states: numpy.ndarray, optional
            The states indices
        grid_data: tuple, optional
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        gdata = grids.get_grid_yz(
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
            verbosity-self.verbosity_delta,
        )
        gdatb = (gdata[1], gdata[2], gdata[0], gdata[3])

        data = self._calc_mean_data(
            "yz",
            data_format,
            variables,
            *gdatb,
            normalize_y,
            normalize_z,
            normalize_x,
            normalize_v,
            label_map,
            vmin,
            vmax,
            states_sel,
            states_isel,
            weight_turbine,
            to_file,
            write_pars,
            ret_states,
            verbosity,
            **kwargs,
        )

        if ret_grid:
            out = list(data) if ret_states else [data]
            return tuple(out + [gdata])
        return data

    def _calc_states_data(
        self,
        ori,
        data_format,
        variables,
        a_pos,
        b_pos,
        c_pos,
        g_pts,
        normalize_a,
        normalize_b,
        normalize_c,
        normalize_v,
        label_map,
        vmin,
        vmax,
        states_sel,
        states_isel,
        to_file,
        write_pars,
        ret_states,
        verbosity,
        **kwargs,
    ):
        """Helper function for states data calculation"""
        # calculate point results:
        if states_sel is not None:
            kwargs["sel"] = {FC.STATE: states_sel}
        if states_isel is not None:
            kwargs["isel"] = {FC.STATE: states_isel}
        point_results = grids.calc_point_results(
            algo=self.algo,
            farm_results=self.fres,
            g_pts=g_pts,
            verbosity=verbosity-self.verbosity_delta,
            **kwargs,
        )
        states = point_results[FC.STATE].to_numpy()
        if variables is None:
            variables = list(point_results.data_vars.keys())
        else:
            point_results.drop_vars(variables)
        del g_pts

        # convert to numpy:
        data = {v: point_results[v].to_numpy() for v in variables}
        del point_results

        # apply data modification:
        a_pos, b_pos, c_pos, data = self._data_mod(
            a_pos,
            b_pos,
            c_pos,
            data,
            normalize_a,
            normalize_b,
            normalize_c,
            normalize_v,
            vmin,
            vmax,
        )

        # translate to selected format:
        if data_format == "numpy":
            data = grids.np2np_sp(data, states, a_pos, b_pos)
            self._write(data_format, data, to_file, verbosity, **write_pars)
        elif data_format == "pandas":
            data = grids.np2pd_sp(data, states, a_pos, b_pos, ori, label_map)
            self._write(data_format, data, to_file, verbosity, **write_pars)
        elif data_format == "xarray":
            data = grids.np2xr_sp(data, states, a_pos, b_pos, c_pos, ori, label_map)
            self._write(data_format, data, to_file, verbosity, **write_pars)
        else:
            raise ValueError(
                f"Unknown data format '{data_format}', choices: numpy, pandas, xarray"
            )

        return (data, states) if ret_states else data

    def get_states_data_xy(
        self,
        resolution,
        variables=None,
        data_format="xarray",
        xmin=None,
        ymin=None,
        xmax=None,
        ymax=None,
        z=None,
        xspace=500.0,
        yspace=500.0,
        normalize_x=None,
        normalize_y=None,
        normalize_z=None,
        normalize_v={},
        label_map={},
        vmin={},
        vmax={},
        states_sel=None,
        states_isel=None,
        to_file=None,
        write_pars={},
        ret_states=False,
        ret_grid=False,
        verbosity=0,
        **kwargs,
    ):
        """
        Creates states data of 2D farm flow slices in a horizontal xy-plane.

        Parameters
        ----------
        resolution: float
            The resolution in m
        variables: list of str, optional
            The variables, or None for all
        data_format: str
            The output data format: numpy, pandas, xarray
        xmin: float, optional
            The min x coordinate, or None for automatic
        ymin: float, optional
            The min y coordinate, or None for automatic
        xmax: float, optional
            The max x coordinate, or None for automatic
        ymax: float, optional
            The max y coordinate, or None for automatic
        z: float, optional
            The z coordinate of the plane
        xspace: float, optional
            The extra space in x direction, before and after wind farm
        yspace: float, optional
            The extra space in y direction, before and after wind farm
        normalize_x: float, optional
            Divide x by this value
        normalize_y: float, optional
            Divide y by this value
        normalize_z: float, optional
            Divide z by this value
        normalize_v: dict, optional
            Divide the variables by these values
        label_map: dict
            The mapping from original to new field names
        vmin: dict
            Minimal values for variables
        vmax: dict
            Maximal values for variables
        states_sel: list, optional
            Reduce to selected states
        states_isel: list, optional
            Reduce to the selected states indices
        to_file: str, optional
            Write data to this file name
        write_pars: dict
            Additional write function parameters
        ret_states: bool
            Flag for returning states indices
        ret_grid: bool
            Flag for returning grid data
        verbosity: int, optional
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Returns
        -------
        data: dict or pandas.DataFrame or xarray.Dataset
            The gridded data
        states: numpy.ndarray, optional
            The states indices
        grid_data: tuple, optional
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        gdata = grids.get_grid_xy(
            self.fres, 
            resolution, 
            xmin, 
            ymin, 
            xmax, 
            ymax, 
            z, 
            xspace, 
            yspace, 
            verbosity-self.verbosity_delta,
        )

        data = self._calc_states_data(
            "xy",
            data_format,
            variables,
            *gdata,
            normalize_x,
            normalize_y,
            normalize_z,
            normalize_v,
            label_map,
            vmin,
            vmax,
            states_sel,
            states_isel,
            to_file,
            write_pars,
            ret_states,
            verbosity,
            **kwargs,
        )

        if ret_grid:
            out = list(data) if ret_states else [data]
            return tuple(out + [gdata])
        return data

    def get_states_data_xz(
        self,
        resolution,
        variables=None,
        data_format="xarray",
        x_direction=270,
        xmin=None,
        zmin=0.0,
        xmax=None,
        zmax=None,
        y=None,
        xspace=500.0,
        zspace=500.0,
        normalize_x=None,
        normalize_y=None,
        normalize_z=None,
        normalize_v={},
        label_map={},
        vmin={},
        vmax={},
        states_sel=None,
        states_isel=None,
        to_file=None,
        write_pars={},
        ret_states=False,
        ret_grid=False,
        verbosity=0,
        **kwargs,
    ):
        """
        Creates states data of 2D farm flow slices in an xz-plane.

        Parameters
        ----------
        resolution: float
            The resolution in m
        variables: list of str, optional
            The variables, or None for all
        data_format: str
            The output data format: numpy, pandas, xarray
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
        y: float, optional
            The y coordinate of the plane
        xspace: float, optional
            The extra space in x direction, before and after wind farm
        zspace: float, optional
            The extra space in z direction, below and above wind farm
        normalize_x: float, optional
            Divide x by this value
        normalize_y: float, optional
            Divide y by this value
        normalize_z: float, optional
            Divide z by this value
        normalize_v: dict, optional
            Divide the variables by these values
        label_map: dict
            The mapping from original to new field names
        vmin: dict
            Minimal values for variables
        vmax: dict
            Maximal values for variables
        states_sel: list, optional
            Reduce to selected states
        states_isel: list, optional
            Reduce to the selected states indices
        to_file: str, optional
            Write data to this file name
        write_pars: dict
            Additional write function parameters
        ret_states: bool
            Flag for returning states indices
        ret_grid: bool
            Flag for returning grid data
        verbosity: int, optional
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Returns
        -------
        data: dict or pandas.DataFrame or xarray.Dataset
            The gridded data
        states: numpy.ndarray, optional
            The states indices
        grid_data: tuple, optional
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        gdata = grids.get_grid_xz(
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
            verbosity-self.verbosity_delta,
        )
        gdatb = (gdata[0], gdata[2], gdata[1], gdata[3])

        data = self._calc_states_data(
            "xz",
            data_format,
            variables,
            *gdatb,
            normalize_x,
            normalize_z,
            normalize_y,
            normalize_v,
            label_map,
            vmin,
            vmax,
            states_sel,
            states_isel,
            to_file,
            write_pars,
            ret_states,
            verbosity,
            **kwargs,
        )

        if ret_grid:
            out = list(data) if ret_states else [data]
            return tuple(out + [gdata])
        return data

    def get_states_data_yz(
        self,
        resolution,
        variables=None,
        data_format="xarray",
        x_direction=270,
        ymin=None,
        zmin=0.0,
        ymax=None,
        zmax=None,
        x=None,
        yspace=500.0,
        zspace=500.0,
        normalize_x=None,
        normalize_y=None,
        normalize_z=None,
        normalize_v={},
        label_map={},
        vmin={},
        vmax={},
        states_sel=None,
        states_isel=None,
        to_file=None,
        write_pars={},
        ret_states=False,
        ret_grid=False,
        verbosity=0,
        **kwargs,
    ):
        """
        Creates states data of 2D farm flow slices in a yz-plane.

        Parameters
        ----------
        resolution: float
            The resolution in m
        variables: list of str, optional
            The variables, or None for all
        data_format: str
            The output data format: numpy, pandas, xarray
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
        x: float, optional
            The x coordinate of the plane
        yspace: float, optional
            The extra space in y direction, before and after wind farm
        zspace: float, optional
            The extra space in z direction, below and above wind farm
        normalize_x: float, optional
            Divide x by this value
        normalize_y: float, optional
            Divide y by this value
        normalize_z: float, optional
            Divide z by this value
        normalize_v: dict, optional
            Divide the variables by these values
        label_map: dict
            The mapping from original to new field names
        vmin: dict
            Minimal values for variables
        vmax: dict
            Maximal values for variables
        states_sel: list, optional
            Reduce to selected states
        states_isel: list, optional
            Reduce to the selected states indices
        to_file: str, optional
            Write data to this file name
        write_pars: dict
            Additional write function parameters
        ret_states: bool
            Flag for returning states indices
        ret_grid: bool
            Flag for returning grid data
        verbosity: int, optional
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Parameters forwarded to the algorithm's calc_points
            function.

        Returns
        -------
        data: dict or pandas.DataFrame or xarray.Dataset
            The gridded data
        states: numpy.ndarray, optional
            The states indices
        grid_data: tuple, optional
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        gdata = grids.get_grid_yz(
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
            verbosity-self.verbosity_delta,
        )
        gdatb = (gdata[1], gdata[2], gdata[0], gdata[3])

        data = self._calc_states_data(
            "yz",
            data_format,
            variables,
            *gdatb,
            normalize_y,
            normalize_z,
            normalize_x,
            normalize_v,
            label_map,
            vmin,
            vmax,
            states_sel,
            states_isel,
            to_file,
            write_pars,
            ret_states,
            verbosity,
            **kwargs,
        )

        if ret_grid:
            out = list(data) if ret_states else [data]
            return tuple(out + [gdata])
        return data
