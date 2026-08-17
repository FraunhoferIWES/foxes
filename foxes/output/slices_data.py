import xarray as xr
import pandas as pd
from typing import Any

import foxes.constants as FC
from foxes.core import Algorithm

from .output import Output
from .slice_data import SliceData


class SlicesData(Output):
    """
    Create data for horizontal or vertical 2D slices, all in a
    single Dataset

    :group: output

    """

    def __init__(
        self,
        algo: Algorithm,
        farm_results: xr.Dataset,
        verbosity_delta: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        algo
            The algorithm for point calculation
        farm_results
            The farm results
        verbosity_delta
            Verbosity threshold for printing calculation info
        kwargs
            Additional parameters for the Output class

        """
        super().__init__(**kwargs)
        self._slice_data = SliceData(
            algo=algo,
            farm_results=farm_results,
            verbosity_delta=verbosity_delta,
            **kwargs,
        )

    def get_mean_data_xy(
        self,
        z_list: list[float],
        verbosity: int = 0,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Creates mean data of 2D farm flow slices in a xy-plane.

        Parameters
        ----------
        z_list
            The z values
        args
            Arguments for the SliceData function of the same name
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Arguments for the SliceData function of the same name

        Returns
        -------
        data
            The gridded data

        """
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"z", "data_format", "ret_states", "ret_grid", "verbosity"}
        }
        dsl = []
        for z in z_list:
            if verbosity > 0:
                print(f"{type(self).__name__}: Creating slice z = {z}")
            dsl.append(
                self._slice_data.get_mean_data_xy(
                    z=z,
                    data_format="xarray",
                    ret_states=False,
                    ret_grid=False,
                    verbosity=verbosity,
                    **kwargs,
                )
            )
        out = xr.concat(dsl, pd.Index(z_list, name="z"))
        del out.attrs["z"]
        return out

    def get_mean_data_xz(
        self,
        y_list: list[float],
        verbosity: int = 0,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Creates mean data of 2D farm flow slices in a xz-plane.

        Parameters
        ----------
        y_list
            The y values
        args
            Arguments for the SliceData function of the same name
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Arguments for the SliceData function of the same name

        Returns
        -------
        data
            The gridded data

        """
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"y", "data_format", "ret_states", "ret_grid", "verbosity"}
        }
        dsl = []
        for y in y_list:
            if verbosity > 0:
                print(f"{type(self).__name__}: Creating slice y = {y}")
            dsl.append(
                self._slice_data.get_mean_data_xz(
                    y=y,
                    data_format="xarray",
                    ret_states=False,
                    ret_grid=False,
                    verbosity=verbosity,
                    **kwargs,
                )
            )
        out = xr.concat(dsl, pd.Index(y_list, name="y"))
        del out.attrs["y"]
        return out

    def get_mean_data_yz(
        self,
        x_list: list[float],
        verbosity: int = 0,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Creates mean data of 2D farm flow slices in a yz-plane.

        Parameters
        ----------
        x_list
            The x values
        args
            Arguments for the SliceData function of the same name
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Arguments for the SliceData function of the same name

        Returns
        -------
        data
            The gridded data

        """
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"x", "data_format", "ret_states", "ret_grid", "verbosity"}
        }
        dsl = []
        for x in x_list:
            if verbosity > 0:
                print(f"{type(self).__name__}: Creating slice x = {x}")
            dsl.append(
                self._slice_data.get_mean_data_yz(
                    x=x,
                    data_format="xarray",
                    ret_states=False,
                    ret_grid=False,
                    verbosity=verbosity,
                    **kwargs,
                )
            )
        out = xr.concat(dsl, pd.Index(x_list, name="x"))
        del out.attrs["x"]
        return out

    def get_states_data_xy(
        self,
        z_list: list[float],
        verbosity: int = 0,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Creates states data of 2D farm flow slices in a xy-plane.

        Parameters
        ----------
        z_list
            The z values
        args
            Arguments for the SliceData function of the same name
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Arguments for the SliceData function of the same name

        Returns
        -------
        data
            The gridded data

        """
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"z", "data_format", "ret_states", "ret_grid", "verbosity"}
        }
        dsl = []
        for z in z_list:
            if verbosity > 0:
                print(f"{type(self).__name__}: Creating slice z = {z}")
            dsl.append(
                self._slice_data.get_states_data_xy(
                    z=z,
                    data_format="xarray",
                    ret_states=False,
                    ret_grid=False,
                    verbosity=verbosity,
                    **kwargs,
                )
            )
        out = xr.concat(dsl, pd.Index(z_list, name="z"))
        del out.attrs["z"]
        return out.transpose(FC.STATE, "z", ...)

    def get_states_data_xz(
        self,
        y_list: list[float],
        verbosity: int = 0,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Creates states data of 2D farm flow slices in a xz-plane.

        Parameters
        ----------
        y_list
            The y values
        args
            Arguments for the SliceData function of the same name
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Arguments for the SliceData function of the same name

        Returns
        -------
        data
            The gridded data

        """
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"y", "data_format", "ret_states", "ret_grid", "verbosity"}
        }
        dsl = []
        for y in y_list:
            if verbosity > 0:
                print(f"{type(self).__name__}: Creating slice y = {y}")
            dsl.append(
                self._slice_data.get_states_data_xz(
                    y=y,
                    data_format="xarray",
                    ret_states=False,
                    ret_grid=False,
                    verbosity=verbosity,
                    **kwargs,
                )
            )
        out = xr.concat(dsl, pd.Index(y_list, name="y"))
        del out.attrs["y"]
        return out.transpose(FC.STATE, "z", "y", ...)

    def get_states_data_yz(
        self,
        x_list: list[float],
        verbosity: int = 0,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Creates states data of 2D farm flow slices in a yz-plane.

        Parameters
        ----------
        x_list
            The x values
        args
            Arguments for the SliceData function of the same name
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Arguments for the SliceData function of the same name

        Returns
        -------
        data
            The gridded data

        """
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"x", "data_format", "ret_states", "ret_grid", "verbosity"}
        }
        dsl = []
        for x in x_list:
            if verbosity > 0:
                print(f"{type(self).__name__}: Creating slice x = {x}")
            dsl.append(
                self._slice_data.get_states_data_yz(
                    x=x,
                    data_format="xarray",
                    ret_states=False,
                    ret_grid=False,
                    verbosity=verbosity,
                    **kwargs,
                )
            )
        out = xr.concat(dsl, pd.Index(x_list, name="x"))
        del out.attrs["x"]
        return out.transpose(FC.STATE, "z", "y", "x", ...)
