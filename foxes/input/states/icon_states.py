from pandas import read_csv

from foxes.data import MODEL_DATA
import foxes.variables as FV

from .field_data import LatLonFieldData


class ICONStates(LatLonFieldData):
    """
    Heterogeneous ambient states in DWD-ICON format.

    :group: input.states

    """

    def __init__(
        self,
        data_source,
        height_coord_default="height",
        height_coord_tke="height_2",
        time_coord="time",
        lat_coord="lat",
        lon_coord="lon",
        output_vars=None,
        var2ncvar=None,
        load_mode="fly",
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source: str
            The input netcdf file(s) containing, can contain
            wildcards, e.g. '2025*_icon.nc'
        height_coord_default: str, optional
            The default height level coordinate name in the data
        height_coord_tke: str, optional
            The height level coordinate name for TKE in the data
        time_coord: str
            The time coordinate name in the data
        lat_coord: str
            The latitude coordinate name in the data
        lon_coord: str
            The longitude coordinate name in the data
        output_vars: list of str, optional
            The output variables to load, if None,
            the default variables are loaded
            (FV.WS, FV.WD, FV.TI, FV.RHO)
        var2ncvar: dict, optional
            A dictionary mapping foxes variable names
            to the corresponding netcdf variable names.
        load_mode: str
            The load mode, choices: preload, lazy, fly.
            preload loads all data during initialization,
            lazy lazy-loads the data using dask, and fly
            reads only states index and weights during initialization
            and then opens the relevant files again within
            the chunk calculations.
        kwargs: dict, optional
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

    def _preproc_icon_nc(self, ds):
        """Preprocess ICON netcdf dataset."""
        if FV.H in self._cmap and self._cmap[FV.H] in ds.sizes:
            c = ds[self._cmap[FV.H]].values.astype(int)
            ds = ds.assign_coords({self._cmap[FV.H]: self.__icon_heights_default[c]})
        if self.H_TKE in self._cmap and self._cmap[self.H_TKE] in ds.sizes:
            c = ds[self._cmap[self.H_TKE]].values.astype(int)
            ds = ds.assign_coords({self._cmap[self.H_TKE]: self.__icon_heights_TKE[c]})
        return self._prepr0(ds) if self._prepr0 is not None else ds

    def load_data(self, algo, verbosity=0):
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        idata: dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

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

        return super().load_data(
            algo,
            bounds_extra_space=self.bounds_extra_space,
            height_bounds=self.height_bounds,
            verbosity=verbosity,
        )

    def _update_dims(self, dims, coords, vrs, d, fdata):
        """Helper function for dimension adjustment, if needed"""
        if self.H_TKE in dims:
            assert FV.H not in dims, (
                f"States {self.name}: Cannot have both {FV.H} and {self.H_TKE} in dims for variables {vrs}, got dims = {dims}"
            )
            dims_new = list(dims)
            dims_new[dims.index(self.H_TKE)] = FV.H
            dims = tuple(dims_new)
            coords = coords.copy()
            coords[FV.H] = coords[self.H_TKE]
        return dims, coords
