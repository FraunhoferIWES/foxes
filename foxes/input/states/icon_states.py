import numpy as np
import xarray as xr
from pandas import read_csv
from pathlib import Path
from typing import Any

from foxes.core import Algorithm, FData, LoadedData, MData

from foxes.data import MODEL_DATA
import foxes.variables as FV

from .field_data import LatLonFieldData


class ICONStates(LatLonFieldData):
    """
    Heterogeneous ambient states in DWD-ICON format.


    """

    def __init__(
        self,
        data_source: str | Path | xr.Dataset,
        height_coord_default: str | None = "height",
        height_coord_tke: str | None = "height_2",
        time_coord: str = "time",
        lat_coord: str = "lat",
        lon_coord: str = "lon",
        output_vars: list[str] | None = None,
        var2ncvar: dict[str, str] | None = None,
        load_mode: str = "fly",
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_source
            The input netcdf file(s) containing, can contain
            wildcards, e.g. '2025*_icon.nc'
        height_coord_default
            The default height level coordinate name in the data
        height_coord_tke
            The height level coordinate name for TKE in the data
        time_coord
            The time coordinate name in the data
        lat_coord
            The latitude coordinate name in the data
        lon_coord
            The longitude coordinate name in the data
        output_vars
            The output variables to load, if None,
            the default variables are loaded
            (FV.WS, FV.WD, FV.TI, FV.RHO)
        var2ncvar
            A dictionary mapping foxes variable names
            to the corresponding netcdf variable names.
        load_mode
            The load mode, choices: preload, lazy, fly.
            preload loads all data during initialization,
            lazy lazy-loads the data using dask, and fly
            reads only states index and weights during initialization
            and then opens the relevant files again within
            the chunk calculations.
        kwargs
            Additional parameters for the base class

        """
        if output_vars is None:
            ovars = [FV.WS, FV.WD, FV.TI, FV.RHO]
        else:
            ovars = output_vars

        if var2ncvar is None:
            var2ncvar = {
                FV.U: "U",
                FV.V: "V",
                FV.TKE: "TKE",
                FV.RHO: "RHO",
            }

        super().__init__(
            data_source=data_source,
            states_coord=time_coord,
            lat_coord=lat_coord,
            lon_coord=lon_coord,
            output_vars=ovars,
            var2ncvar=var2ncvar,
            load_mode=load_mode,
            **kwargs,
        )

        self._prepr0 = self.preprocess_nc
        self.preprocess_nc = self._preproc_icon_nc

        self.variables = []
        for v in ovars:
            if v == FV.TI:
                self.variables.append(FV.TKE)
            elif v == FV.WS or v == FV.WD:
                if FV.U not in self.variables:
                    assert FV.WS in ovars and FV.WD in ovars, (
                        f"{self.name}: Both FV.WS and FV.WD must be requested if one of them is requested."
                    )
                    self.variables.extend([FV.U, FV.V])
            elif v == FV.RHO:
                self.variables.append(FV.T)
                self.variables.append(FV.p)
            elif v not in self.variables:
                self.variables.append(v)

        # adjust height coordinate mapping for ICON data:
        if height_coord_default is not None:
            self._cmap[FV.H] = height_coord_default
        if height_coord_tke is not None:
            self.H_TKE = FV.H + "_tke"
            self._cmap[self.H_TKE] = height_coord_tke

    def _preproc_icon_nc(self, ds: xr.Dataset) -> xr.Dataset:
        """Preprocess ICON netcdf dataset.

        Parameters
        ----------
        ds
            The dataset to preprocess.

        Returns
        -------
        dataset
            The preprocessed dataset.

        """
        if FV.H in self._cmap and self._cmap[FV.H] in ds.sizes:
            c = ds[self._cmap[FV.H]].values.astype(int)
            ds = ds.assign_coords({self._cmap[FV.H]: self.__icon_heights_default[c]})
        if self.H_TKE in self._cmap and self._cmap[self.H_TKE] in ds.sizes:
            c = ds[self._cmap[self.H_TKE]].values.astype(int)
            ds = ds.assign_coords({self._cmap[self.H_TKE]: self.__icon_heights_TKE[c]})
        return self._prepr0(ds) if self._prepr0 is not None else ds

    def load_data(  # type: ignore[override]
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        bounds_extra_space: float | str | None = None,
        height_bounds: tuple[float, float] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        bounds_extra_space
            The extra space to add to the horizontal wind farm bounds.
        height_bounds
            The height bounds in m.
        verbosity
            The verbosity level, 0 = silent

        """
        # read mapping from height levels to heights from static csv files:
        hdata = {}
        for A in ("A1", "A2"):
            fpath = algo.dbook.get_file_path(
                MODEL_DATA, f"icon_heights_{A}.csv", check_raw=False
            )
            algo.print(f"States '{self.name}': Loading '{fpath.stem}'")
            hdata[A] = read_csv(fpath, index_col="eu_nest")["height"].to_numpy()
        self.__icon_heights_TKE = hdata["A1"]
        self.__icon_heights_default = hdata["A2"]

        super().load_data(
            algo,
            loaded_data,
            force=force,
            bounds_extra_space=self.bounds_extra_space,
            height_bounds=self.height_bounds,
            verbosity=verbosity,
        )

    def _get_calc_data(
        self, mdata: MData, fdata: FData
    ) -> tuple[dict[tuple[str, ...], tuple[list[str], np.ndarray]], np.ndarray | None]:
        """
        Gathers data for calculations.

        Parameters
        ----------
        mdata
            The mdata object
        fdata
            The fdata object

        Returns
        -------
        data
            The extracted data, keys are dimension tuples,
            values are tuples (variables, data_array)
            where variables contains variable names, and
            data_array contains the data values,
            the last dimension corresponds to the variables
        weights
            The weights array, if only state dependent, otherwise
            weights are among data. Shape: (n_states,)

        """
        data, weights = super()._get_calc_data(mdata, fdata)
        dmlst = list(data.keys())
        for dims in dmlst:
            if self.H_TKE in dims:
                vrs, d = data.pop(dims)
                assert FV.H not in dims, (
                    f"States {self.name}: Cannot have both {FV.H} and {self.H_TKE} in dims for variables {vrs}, got dims = {dims}"
                )
                dims_new = tuple(FV.H if dim == self.H_TKE else dim for dim in dims)
                data[dims_new] = (vrs, d)
        return data, weights

    def get_interpolation_coords(
        self, mdata: MData, idims: list[str]
    ) -> dict[str, np.ndarray]:
        """
        Extracts interpolation coordinates from chunk model data.

        Parameters
        ----------
        mdata
            The model data
        idims
            The input dimensions, e.g. [x, y, height]

        Returns
        -------
        icoords
            The extracted interpolation coordinates, keys are dimension names,
            values are one-dimensional arrays with the coordinate values

        """
        icoords = super().get_interpolation_coords(mdata, idims)  # type: ignore[misc]
        if self.H_TKE in idims:
            assert FV.H not in idims, (
                f"States {self.name}: Cannot have both {FV.H} and {self.H_TKE} in idims for interpolation, got idims = {idims}"
            )
            icoords[FV.H] = icoords.pop(self.H_TKE)
        return icoords
