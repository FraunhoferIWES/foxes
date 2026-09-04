from __future__ import annotations

import numpy as np
from xarray import Dataset
from typing import TYPE_CHECKING, Any

from foxes.config import config
from foxes.utils import write_nc
import foxes.variables as FV
import foxes.constants as FC

from .output import Output
from . import grids

if TYPE_CHECKING:
    from foxes.core import Algorithm


class SliceData(Output):
    """
    Create data for horizontal or vertical 2D slices
    """

    def __init__(
        self,
        algo: Algorithm,
        farm_results: Dataset,
        verbosity_delta: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        algo
            The algorithm for point calculation
        farm_results
            The farm results
        verbosity_delta
            Verbosity threshold for printing calculation info
        kwargs
            Additional parameters for the base class
        """
        super().__init__(**kwargs)
        self.algo = algo
        self.fres = farm_results
        self.verbosity_delta = verbosity_delta

    def _data_mod(
        self,
        a_pos: np.ndarray,
        b_pos: np.ndarray,
        c_pos: float,
        data: Any,
        normalize_a: float | None,
        normalize_b: float | None,
        normalize_c: float | None,
        normalize_v: dict[str, float],
        vmin: dict[str, Any],
        vmax: dict[str, Any],
    ) -> tuple[Any, Any, Any, Any]:
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

    def _write(
        self,
        format: str,
        data: Any,
        fname: str | None,
        verbosity: int,
        **write_pars: Any,
    ) -> None:
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
                fpath = self.get_fpath(fname)
                if verbosity > 0:
                    print("Writing file", fpath)
                self.write(fname, data, **write_pars)

            elif format == "xarray":
                nc_engine = config.nc_engine
                assert nc_engine is not None
                write_nc(
                    data,
                    self.get_fpath(fname),
                    nc_engine=nc_engine,
                    verbosity=verbosity,
                    **write_pars,
                )

            else:
                raise ValueError(
                    f"Unknown data format '{format}', choices: numpy, pandas, xarray"
                )

    def _calc_mean_data(
        self,
        ori: str,
        data_format: str,
        variables: list[str] | None,
        a_pos: np.ndarray,
        b_pos: np.ndarray,
        c_pos: float,
        g_pts: np.ndarray,
        normalize_a: float | None,
        normalize_b: float | None,
        normalize_c: float | None,
        normalize_v: dict[str, float],
        label_map: dict[str, str],
        vmin: dict[str, Any],
        vmax: dict[str, Any],
        to_file: str | None,
        write_pars: dict[str, Any],
        ret_states: bool,
        verbosity: int,
        **kwargs: Any,
    ) -> Any:
        """Helper function for mean data calculation"""
        # calculate point results:
        point_results = grids.calc_point_results(
            algo=self.algo,
            farm_results=self.fres,
            g_pts=g_pts,
            verbosity=verbosity - self.verbosity_delta,
            **kwargs,
        )
        states = point_results[FC.STATE].to_numpy()
        if variables is None:
            variables = list(point_results.data_vars.keys())
        else:
            point_results.drop_vars(variables)
        del g_pts

        # take mean over states:
        weights = point_results[FV.WEIGHT].to_numpy()
        data: Any
        if point_results[FV.WEIGHT].dims == (FC.STATE,):
            data = {
                v: np.einsum("s,sp->p", weights, point_results[v].to_numpy())
                for v in variables
            }
        elif point_results[FV.WEIGHT].dims == (FC.STATE, FC.POINT):
            data = {
                v: np.einsum("sp,sp->p", weights, point_results[v].to_numpy())
                for v in variables
            }
        else:
            raise ValueError(
                f"Wrong dimensions for '{FV.WEIGHT}': Expecting {(FC.STATE,)} or {(FC.STATE, FC.POINT)}, got {point_results[FV.WEIGHT].dims}"
            )
        del point_results, weights

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
        resolution: Any = None,
        n_img_points: Any = None,
        variables: list[str] | None = None,
        data_format: str = "xarray",
        xmin: float | None = None,
        ymin: float | None = None,
        xmax: float | None = None,
        ymax: float | None = None,
        z: float | None = None,
        xspace: float = 500.0,
        yspace: float = 500.0,
        normalize_x: float | None = None,
        normalize_y: float | None = None,
        normalize_z: float | None = None,
        normalize_v: dict[str, float] = {},
        label_map: dict[str, str] = {},
        vmin: dict[str, Any] = {},
        vmax: dict[str, Any] = {},
        states_sel: Any = None,
        states_isel: Any = None,
        to_file: str | None = None,
        write_pars: dict[str, Any] = {},
        ret_states: bool = False,
        ret_grid: bool = False,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> Any:
        """
        Creates mean data of 2D farm flow slices in a horizontal xy-plane.

        Parameters
        ----------
        resolution
            The resolution in m
        n_img_points
            The number of image points (n, m) in the two directions
        variables
            The variables, or None for all
        data_format
            The output data format: numpy, pandas, xarray
        xmin
            The min x coordinate, or None for automatic
        ymin
            The min y coordinate, or None for automatic
        xmax
            The max x coordinate, or None for automatic
        ymax
            The max y coordinate, or None for automatic
        z
            The z coordinate of the plane
        xspace
            The extra space in x direction, before and after wind farm
        yspace
            The extra space in y direction, before and after wind farm
        normalize_x
            Divide x by this value
        normalize_y
            Divide y by this value
        normalize_z
            Divide z by this value
        normalize_v
            Divide the variables by these values
        label_map
            The mapping from original to new field names
        vmin
            Minimal values for variables
        vmax
            Maximal values for variables
        states_sel
            Reduce to selected states
        states_isel
            Reduce to the selected states indices
        to_file
            Write data to this file name
        write_pars
            Additional write function parameters
        ret_states
            Flag for returning states indices
        ret_grid
            Flag for returning grid data
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Parameters forwarded to the algorithm's calc_points
            function.

        Returns
        -------
        data
            The gridded data
        states
            The states indices
        grid_data
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        gdata = grids.get_grid_xy(
            farm_results=self.fres,
            resolution=resolution,
            n_img_points=n_img_points,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            z=z,
            xspace=xspace,
            yspace=yspace,
            states_sel=states_sel,
            states_isel=states_isel,
            verbosity=verbosity - self.verbosity_delta,
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
            to_file,
            write_pars,
            ret_states,
            verbosity,
            states_sel=states_sel,
            states_isel=states_isel,
            **kwargs,
        )

        if ret_grid:
            out = list(data) if ret_states else [data]
            return tuple(out + [gdata])
        return data

    def get_mean_data_xz(
        self,
        resolution: Any = None,
        n_img_points: Any = None,
        variables: list[str] | None = None,
        data_format: str = "xarray",
        x_direction: float = 270,
        xmin: float | None = None,
        zmin: float = 0.0,
        xmax: float | None = None,
        zmax: Any = None,
        y: float | None = None,
        xspace: float = 500.0,
        zspace: float = 500.0,
        normalize_x: float | None = None,
        normalize_y: float | None = None,
        normalize_z: float | None = None,
        normalize_v: dict[str, float] = {},
        label_map: dict[str, str] = {},
        vmin: dict[str, Any] = {},
        vmax: dict[str, Any] = {},
        states_sel: Any = None,
        states_isel: Any = None,
        to_file: str | None = None,
        write_pars: dict[str, Any] = {},
        ret_states: bool = False,
        ret_grid: bool = False,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> Any:
        """
        Creates mean data of 2D farm flow slices in an xz-plane.

        Parameters
        ----------
        resolution
            The resolution in m
        n_img_points
            The number of image points (n, m) in the two directions
        variables
            The variables, or None for all
        data_format
            The output data format: numpy, pandas, xarray
        x_direction
            The direction of the x axis, 0 = north
        xmin
            The min x coordinate, or None for automatic
        zmin
            The min z coordinate
        xmax
            The max x coordinate, or None for automatic
        zmax
            The max z coordinate, or None for automatic
        y
            The y coordinate of the plane
        xspace
            The extra space in x direction, before and after wind farm
        zspace
            The extra space in z direction, below and above wind farm
        normalize_x
            Divide x by this value
        normalize_y
            Divide y by this value
        normalize_z
            Divide z by this value
        normalize_v
            Divide the variables by these values
        label_map
            The mapping from original to new field names
        vmin
            Minimal values for variables
        vmax
            Maximal values for variables
        states_sel
            Reduce to selected states
        states_isel
            Reduce to the selected states indices
        to_file
            Write data to this file name
        write_pars
            Additional write function parameters
        ret_states
            Flag for returning states indices
        ret_grid
            Flag for returning grid data
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Parameters forwarded to the algorithm's calc_points
            function.

        Returns
        -------
        data
            The gridded data
        states
            The states indices
        grid_data
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        gdata = grids.get_grid_xz(
            farm_results=self.fres,
            resolution=resolution,
            n_img_points=n_img_points,
            x_direction=x_direction,
            xmin=xmin,
            zmin=zmin,
            xmax=xmax,
            zmax=zmax,
            y=y,
            xspace=xspace,
            zspace=zspace,
            states_sel=states_sel,
            states_isel=states_isel,
            verbosity=verbosity - self.verbosity_delta,
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
            to_file,
            write_pars,
            ret_states,
            verbosity,
            states_sel=states_sel,
            states_isel=states_isel,
            **kwargs,
        )

        if ret_grid:
            out = list(data) if ret_states else [data]
            return tuple(out + [gdata])
        return data

    def get_mean_data_yz(
        self,
        resolution: Any = None,
        n_img_points: Any = None,
        variables: list[str] | None = None,
        data_format: str = "xarray",
        x_direction: float = 270,
        ymin: float | None = None,
        zmin: float = 0.0,
        ymax: float | None = None,
        zmax: Any = None,
        x: float | None = None,
        yspace: float = 500.0,
        zspace: float = 500.0,
        normalize_x: float | None = None,
        normalize_y: float | None = None,
        normalize_z: float | None = None,
        normalize_v: dict[str, float] = {},
        label_map: dict[str, str] = {},
        vmin: dict[str, Any] = {},
        vmax: dict[str, Any] = {},
        states_sel: Any = None,
        states_isel: Any = None,
        to_file: str | None = None,
        write_pars: dict[str, Any] = {},
        ret_states: bool = False,
        ret_grid: bool = False,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> Any:
        """
        Creates mean data of 2D farm flow slices in a yz-plane.

        Parameters
        ----------
        resolution
            The resolution in m
        n_img_points
            The number of image points (n, m) in the two directions
        variables
            The variables, or None for all
        data_format
            The output data format: numpy, pandas, xarray
        x_direction
            The direction of the x axis, 0 = north
        ymin
            The min y coordinate, or None for automatic
        zmin
            The min z coordinate
        ymax
            The max y coordinate, or None for automatic
        zmax
            The max z coordinate, or None for automatic
        x
            The x coordinate of the plane
        yspace
            The extra space in y direction, before and after wind farm
        zspace
            The extra space in z direction, below and above wind farm
        normalize_x
            Divide x by this value
        normalize_y
            Divide y by this value
        normalize_z
            Divide z by this value
        normalize_v
            Divide the variables by these values
        label_map
            The mapping from original to new field names
        vmin
            Minimal values for variables
        vmax
            Maximal values for variables
        states_sel
            Reduce to selected states
        states_isel
            Reduce to the selected states indices
        to_file
            Write data to this file name
        write_pars
            Additional write function parameters
        ret_states
            Flag for returning states indices
        ret_grid
            Flag for returning grid data
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Parameters forwarded to the algorithm's calc_points
            function.

        Returns
        -------
        data
            The gridded data
        states
            The states indices
        grid_data
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        gdata = grids.get_grid_yz(
            farm_results=self.fres,
            resolution=resolution,
            n_img_points=n_img_points,
            x_direction=x_direction,
            ymin=ymin,
            zmin=zmin,
            ymax=ymax,
            zmax=zmax,
            x=x,
            yspace=yspace,
            zspace=zspace,
            states_sel=states_sel,
            states_isel=states_isel,
            verbosity=verbosity - self.verbosity_delta,
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
            to_file,
            write_pars,
            ret_states,
            verbosity,
            states_sel=states_sel,
            states_isel=states_isel,
            **kwargs,
        )

        if ret_grid:
            out = list(data) if ret_states else [data]
            return tuple(out + [gdata])
        return data

    def _calc_states_data(
        self,
        ori: str,
        data_format: str,
        variables: list[str] | None,
        a_pos: np.ndarray,
        b_pos: np.ndarray,
        c_pos: float,
        g_pts: np.ndarray,
        normalize_a: float | None,
        normalize_b: float | None,
        normalize_c: float | None,
        normalize_v: dict[str, float],
        label_map: dict[str, str],
        vmin: dict[str, Any],
        vmax: dict[str, Any],
        to_file: str | None,
        write_pars: dict[str, Any],
        ret_states: bool,
        verbosity: int,
        **kwargs: Any,
    ) -> Any:
        """Helper function for states data calculation"""
        # calculate point results:
        point_results = grids.calc_point_results(
            algo=self.algo,
            farm_results=self.fres,
            g_pts=g_pts,
            verbosity=verbosity - self.verbosity_delta,
            **kwargs,
        )
        states = point_results[FC.STATE].to_numpy()
        if variables is None:
            variables = list(point_results.data_vars.keys())
        else:
            point_results.drop_vars(variables)
        del g_pts

        # convert to numpy:
        data: Any = {v: point_results[v].to_numpy() for v in variables}
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
        resolution: Any = None,
        n_img_points: Any = None,
        variables: list[str] | None = None,
        data_format: str = "xarray",
        xmin: float | None = None,
        ymin: float | None = None,
        xmax: float | None = None,
        ymax: float | None = None,
        z: float | None = None,
        xspace: float = 500.0,
        yspace: float = 500.0,
        normalize_x: float | None = None,
        normalize_y: float | None = None,
        normalize_z: float | None = None,
        normalize_v: dict[str, float] = {},
        label_map: dict[str, str] = {},
        vmin: dict[str, Any] = {},
        vmax: dict[str, Any] = {},
        states_sel: Any = None,
        states_isel: Any = None,
        to_file: str | None = None,
        write_pars: dict[str, Any] = {},
        ret_states: bool = False,
        ret_grid: bool = False,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> Any:
        """
        Creates states data of 2D farm flow slices in a horizontal xy-plane.

        Parameters
        ----------
        resolution
            The resolution in m
        n_img_points
            The number of image points (n, m) in the two directions
        variables
            The variables, or None for all
        data_format
            The output data format: numpy, pandas, xarray
        xmin
            The min x coordinate, or None for automatic
        ymin
            The min y coordinate, or None for automatic
        xmax
            The max x coordinate, or None for automatic
        ymax
            The max y coordinate, or None for automatic
        z
            The z coordinate of the plane
        xspace
            The extra space in x direction, before and after wind farm
        yspace
            The extra space in y direction, before and after wind farm
        normalize_x
            Divide x by this value
        normalize_y
            Divide y by this value
        normalize_z
            Divide z by this value
        normalize_v
            Divide the variables by these values
        label_map
            The mapping from original to new field names
        vmin
            Minimal values for variables
        vmax
            Maximal values for variables
        states_sel
            Reduce to selected states
        states_isel
            Reduce to the selected states indices
        to_file
            Write data to this file name
        write_pars
            Additional write function parameters
        ret_states
            Flag for returning states indices
        ret_grid
            Flag for returning grid data
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Parameters forwarded to the algorithm's calc_points
            function.

        Returns
        -------
        data
            The gridded data
        states
            The states indices
        grid_data
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        gdata = grids.get_grid_xy(
            farm_results=self.fres,
            resolution=resolution,
            n_img_points=n_img_points,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            z=z,
            xspace=xspace,
            yspace=yspace,
            states_sel=states_sel,
            states_isel=states_isel,
            verbosity=verbosity - self.verbosity_delta,
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
            to_file,
            write_pars,
            ret_states,
            verbosity,
            states_sel=states_sel,
            states_isel=states_isel,
            **kwargs,
        )

        if ret_grid:
            out = list(data) if ret_states else [data]
            return tuple(out + [gdata])
        return data

    def get_states_data_xz(
        self,
        resolution: Any = None,
        n_img_points: Any = None,
        variables: list[str] | None = None,
        data_format: str = "xarray",
        x_direction: float = 270,
        xmin: float | None = None,
        zmin: float = 0.0,
        xmax: float | None = None,
        zmax: Any = None,
        y: float | None = None,
        xspace: float = 500.0,
        zspace: float = 500.0,
        normalize_x: float | None = None,
        normalize_y: float | None = None,
        normalize_z: float | None = None,
        normalize_v: dict[str, float] = {},
        label_map: dict[str, str] = {},
        vmin: dict[str, Any] = {},
        vmax: dict[str, Any] = {},
        states_sel: Any = None,
        states_isel: Any = None,
        to_file: str | None = None,
        write_pars: dict[str, Any] = {},
        ret_states: bool = False,
        ret_grid: bool = False,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> Any:
        """
        Creates states data of 2D farm flow slices in an xz-plane.

        Parameters
        ----------
        resolution
            The resolution in m
        n_img_points
            The number of image points (n, m) in the two directions
        variables
            The variables, or None for all
        data_format
            The output data format: numpy, pandas, xarray
        x_direction
            The direction of the x axis, 0 = north
        xmin
            The min x coordinate, or None for automatic
        zmin
            The min z coordinate
        xmax
            The max x coordinate, or None for automatic
        zmax
            The max z coordinate, or None for automatic
        y
            The y coordinate of the plane
        xspace
            The extra space in x direction, before and after wind farm
        zspace
            The extra space in z direction, below and above wind farm
        normalize_x
            Divide x by this value
        normalize_y
            Divide y by this value
        normalize_z
            Divide z by this value
        normalize_v
            Divide the variables by these values
        label_map
            The mapping from original to new field names
        vmin
            Minimal values for variables
        vmax
            Maximal values for variables
        states_sel
            Reduce to selected states
        states_isel
            Reduce to the selected states indices
        to_file
            Write data to this file name
        write_pars
            Additional write function parameters
        ret_states
            Flag for returning states indices
        ret_grid
            Flag for returning grid data
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Parameters forwarded to the algorithm's calc_points
            function.

        Returns
        -------
        data
            The gridded data
        states
            The states indices
        grid_data
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        gdata = grids.get_grid_xz(
            farm_results=self.fres,
            resolution=resolution,
            n_img_points=n_img_points,
            x_direction=x_direction,
            xmin=xmin,
            zmin=zmin,
            xmax=xmax,
            zmax=zmax,
            y=y,
            xspace=xspace,
            zspace=zspace,
            states_sel=states_sel,
            states_isel=states_isel,
            verbosity=verbosity - self.verbosity_delta,
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
            to_file,
            write_pars,
            ret_states,
            verbosity,
            states_sel=states_sel,
            states_isel=states_isel,
            **kwargs,
        )

        if ret_grid:
            out = list(data) if ret_states else [data]
            return tuple(out + [gdata])
        return data

    def get_states_data_yz(
        self,
        resolution: Any = None,
        n_img_points: Any = None,
        variables: list[str] | None = None,
        data_format: str = "xarray",
        x_direction: float = 270,
        ymin: float | None = None,
        zmin: float = 0.0,
        ymax: float | None = None,
        zmax: Any = None,
        x: float | None = None,
        yspace: float = 500.0,
        zspace: float = 500.0,
        normalize_x: float | None = None,
        normalize_y: float | None = None,
        normalize_z: float | None = None,
        normalize_v: dict[str, float] = {},
        label_map: dict[str, str] = {},
        vmin: dict[str, Any] = {},
        vmax: dict[str, Any] = {},
        states_sel: Any = None,
        states_isel: Any = None,
        to_file: str | None = None,
        write_pars: dict[str, Any] = {},
        ret_states: bool = False,
        ret_grid: bool = False,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> Any:
        """
        Creates states data of 2D farm flow slices in a yz-plane.

        Parameters
        ----------
        resolution
            The resolution in m
        n_img_points
            The number of image points (n, m) in the two directions
        variables
            The variables, or None for all
        data_format
            The output data format: numpy, pandas, xarray
        x_direction
            The direction of the x axis, 0 = north
        ymin
            The min y coordinate, or None for automatic
        zmin
            The min z coordinate
        ymax
            The max y coordinate, or None for automatic
        zmax
            The max z coordinate, or None for automatic
        x
            The x coordinate of the plane
        yspace
            The extra space in y direction, before and after wind farm
        zspace
            The extra space in z direction, below and above wind farm
        normalize_x
            Divide x by this value
        normalize_y
            Divide y by this value
        normalize_z
            Divide z by this value
        normalize_v
            Divide the variables by these values
        label_map
            The mapping from original to new field names
        vmin
            Minimal values for variables
        vmax
            Maximal values for variables
        states_sel
            Reduce to selected states
        states_isel
            Reduce to the selected states indices
        to_file
            Write data to this file name
        write_pars
            Additional write function parameters
        ret_states
            Flag for returning states indices
        ret_grid
            Flag for returning grid data
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Parameters forwarded to the algorithm's calc_points
            function.

        Returns
        -------
        data
            The gridded data
        states
            The states indices
        grid_data
            The grid data (x_pos, y_pos, z_pos, g_pts)

        """
        gdata = grids.get_grid_yz(
            farm_results=self.fres,
            resolution=resolution,
            n_img_points=n_img_points,
            x_direction=x_direction,
            ymin=ymin,
            zmin=zmin,
            ymax=ymax,
            zmax=zmax,
            x=x,
            yspace=yspace,
            zspace=zspace,
            states_sel=states_sel,
            states_isel=states_isel,
            verbosity=verbosity - self.verbosity_delta,
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
            to_file,
            write_pars,
            ret_states,
            verbosity,
            states_sel=states_sel,
            states_isel=states_isel,
            **kwargs,
        )

        if ret_grid:
            out = list(data) if ret_states else [data]
            return tuple(out + [gdata])
        return data
