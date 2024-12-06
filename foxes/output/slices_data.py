import xarray as xr
import pandas as pd

import foxes.constants as FC

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
        algo,
        farm_results,
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
        verbosity_delta: int
            Verbosity threshold for printing calculation info
        kwargs: dict, optional
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
        z_list,
        *args,
        verbosity=0,
        **kwargs,
    ):
        """
        Creates mean data of 2D farm flow slices in a xy-plane.

        Parameters
        ----------
        z_list: list of float
            The z values
        args: tuple, optional
            Arguments for the SliceData function of the same name
        verbosity: int, optional
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Arguments for the SliceData function of the same name

        Returns
        -------
        data: xarray.Dataset
            The gridded data

        """
        dsl = []
        for z in z_list:
            if verbosity > 0:
                print(f"{type(self).__name__}: Creating slice z = {z}")
            dsl.append(
                self._slice_data.get_mean_data_xy(
                    *args,
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
        y_list,
        *args,
        verbosity=0,
        **kwargs,
    ):
        """
        Creates mean data of 2D farm flow slices in a xz-plane.

        Parameters
        ----------
        y_list: list of float
            The y values
        args: tuple, optional
            Arguments for the SliceData function of the same name
        verbosity: int, optional
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Arguments for the SliceData function of the same name

        Returns
        -------
        data: xarray.Dataset
            The gridded data

        """
        dsl = []
        for y in y_list:
            if verbosity > 0:
                print(f"{type(self).__name__}: Creating slice y = {y}")
            dsl.append(
                self._slice_data.get_mean_data_xz(
                    *args,
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
        x_list,
        *args,
        verbosity=0,
        **kwargs,
    ):
        """
        Creates mean data of 2D farm flow slices in a yz-plane.

        Parameters
        ----------
        x_list: list of float
            The x values
        args: tuple, optional
            Arguments for the SliceData function of the same name
        verbosity: int, optional
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Arguments for the SliceData function of the same name

        Returns
        -------
        data: xarray.Dataset
            The gridded data

        """
        dsl = []
        for x in x_list:
            if verbosity > 0:
                print(f"{type(self).__name__}: Creating slice x = {x}")
            dsl.append(
                self._slice_data.get_mean_data_yz(
                    *args,
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
        z_list,
        *args,
        verbosity=0,
        **kwargs,
    ):
        """
        Creates states data of 2D farm flow slices in a xy-plane.

        Parameters
        ----------
        z_list: list of float
            The z values
        args: tuple, optional
            Arguments for the SliceData function of the same name
        verbosity: int, optional
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Arguments for the SliceData function of the same name

        Returns
        -------
        data: xarray.Dataset
            The gridded data

        """
        dsl = []
        for z in z_list:
            if verbosity > 0:
                print(f"{type(self).__name__}: Creating slice z = {z}")
            dsl.append(
                self._slice_data.get_states_data_xy(
                    *args,
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
        y_list,
        *args,
        verbosity=0,
        **kwargs,
    ):
        """
        Creates states data of 2D farm flow slices in a xz-plane.

        Parameters
        ----------
        y_list: list of float
            The y values
        args: tuple, optional
            Arguments for the SliceData function of the same name
        verbosity: int, optional
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Arguments for the SliceData function of the same name

        Returns
        -------
        data: xarray.Dataset
            The gridded data

        """
        dsl = []
        for y in y_list:
            if verbosity > 0:
                print(f"{type(self).__name__}: Creating slice y = {y}")
            dsl.append(
                self._slice_data.get_states_data_xz(
                    *args,
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
        x_list,
        *args,
        verbosity=0,
        **kwargs,
    ):
        """
        Creates states data of 2D farm flow slices in a yz-plane.

        Parameters
        ----------
        x_list: list of float
            The x values
        args: tuple, optional
            Arguments for the SliceData function of the same name
        verbosity: int, optional
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Arguments for the SliceData function of the same name

        Returns
        -------
        data: xarray.Dataset
            The gridded data

        """
        dsl = []
        for x in x_list:
            if verbosity > 0:
                print(f"{type(self).__name__}: Creating slice x = {x}")
            dsl.append(
                self._slice_data.get_states_data_yz(
                    *args,
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
