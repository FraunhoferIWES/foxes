from __future__ import annotations

import numpy as np
from pandas import DataFrame
from xarray import Dataset, open_dataset
from pathlib import Path
from typing import cast

from foxes.core import Algorithm, FData, LoadedData, MData, States, TData
from foxes.config import config, get_input_path
from foxes.data import STATES
import foxes.variables as FV
import foxes.constants as FC

from .dataset_states import DatasetSelection, DatasetStates, InterpolationParameters


class SingleStateField(States):
    """
    Single state field, i.e. no state coordinate, just a regular grid.

    Attributes
    ----------
    data_source: xarray.Dataset or str or pathlib.Path
        The NetCDF dataset to read from, or a path to it.
    output_vars: list of str
        List of variable names to read.
    var2ncvar: dict[str, str]
        Mapping from variable names to netCDF variable names.
    fixed_vars: dict[str, float]
        Mapping from variable names to fixed values.
    x_coord: str
        Name of the x coordinate.
    y_coord: str
        Name of the y coordinate.
    h_coord: str
        Name of the height coordinate.
    sel: dict[str, object]
        Subset selection via xr.Dataset.sel()
    isel: dict[str, object]
        Subset selection via xr.Dataset.isel()
    interp_pars: dict[str, bool or float or str or None]
        Interpolation parameters, passed to the interpolation function.
    bounds_extra_space: float or str
        The extra space, either float in m,
        or str for units of D, e.g. '2.5D'
    height_bounds: tuple[float, float]
        The (h_min, h_max) height bounds in m. Defaults to H +/- 0.5*D

    :group: input.states

    """

    def __init__(
        self,
        data_source: str | Path | Dataset,
        output_vars: list[str],
        var2ncvar: dict[str, str] | None = None,
        fixed_vars: dict[str, float] | None = None,
        x_coord: str = "x",
        y_coord: str = "y",
        h_coord: str | None = "height",
        sel: DatasetSelection | None = None,
        isel: DatasetSelection | None = None,
        interp_pars: InterpolationParameters | None = None,
        bounds_extra_space: float | str | None = 1000,
        height_bounds: tuple[float, float] | None = None,
        **kwargs: object,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_source: xarray.Dataset or str or pathlib.Path
            The NetCDF dataset to read from, or a path to it.
        output_vars: list of str
            List of variable names to read.
        var2ncvar: dict[str, str], optional
            Mapping from variable names to netCDF variable names.
        fixed_vars: dict[str, float], optional
            Mapping from variable names to fixed values.
        x_coord: str
            Name of the x coordinate.
        y_coord: str
            Name of the y coordinate.
        h_coord: str or None
            Name of the height coordinate.
        sel: dict[str, object], optional
            Subset selection via xr.Dataset.sel()
        isel: dict[str, object], optional
            Subset selection via xr.Dataset.isel()
        interp_pars: dict[str, bool or float or str or None], optional
            Interpolation parameters, passed to the interpolation function.
        bounds_extra_space: float or str or None, optional
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        height_bounds: tuple[float, float], optional
            The (h_min, h_max) height bounds in m. Defaults to H +/- 0.5*D
        kwargs: object
            Keyword arguments passed to the base class.

        """
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.data_source = data_source
        self.output_vars = output_vars
        self.var2ncvar = {} if var2ncvar is None else var2ncvar
        self.fixed_vars = {} if fixed_vars is None else fixed_vars
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.h_coord = h_coord
        self.sel = sel
        self.isel = isel
        self.interp_pars = {} if interp_pars is None else interp_pars
        self.bounds_extra_space = bounds_extra_space
        self.height_bounds = height_bounds

        self._cmap = {
            FV.X: self.x_coord,
            FV.Y: self.y_coord,
        }
        if self.h_coord is not None:
            self._cmap[FV.H] = self.h_coord

        self._data: Dataset | None = None

    def output_point_vars(self, algo: Algorithm) -> list[str]:
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars: list of str
            The output variable names

        """
        return self.output_vars

    @property
    def data(self) -> Dataset | None:
        """
        The field data

        Returns
        -------
        d: xrarray.Dataset
            The field data

        """
        return self._data

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 1,
    ) -> None:
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        loaded_data: LoadedData
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force: bool
            Overwrite existing data
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)

        self.DATA = self.var("data")
        if self.DATA not in loaded_data["extra_data"] or force:
            # read NetCDF data file, if not given as Dataset already:
            if isinstance(self.data_source, Dataset):
                data = self.data_source
            else:
                fpath = get_input_path(self.data_source)
                if not fpath.is_file():
                    if algo is not None:
                        if verbosity > 0:
                            print(
                                f"States '{self.name}': Reading static data '{fpath.name}' from context '{STATES}'"
                            )
                        fpath = algo.dbook.get_file_path(
                            STATES, fpath.name, check_raw=False
                        )
                    else:
                        raise FileNotFoundError(
                            f"States '{self.name}': File {fpath} not found."
                        )
                    if verbosity > 0:
                        print(f"Path: {fpath}")
                elif verbosity:
                    print(f"States '{self.name}': Reading file {fpath}")
                data = open_dataset(fpath, engine=config.nc_engine)

            # remove unnecessary variables:
            vrs = {
                var: self.var2ncvar.get(var, var)
                for var in self.output_vars
                if var not in self.fixed_vars
            }
            try:
                data = data[list(vrs.values())]
            except KeyError as e:
                raise KeyError(
                    f"States '{self.name}': Variable '{e.args[0]}' not found in dataset {fpath.name}."
                )

            # check coordinates:
            for c in self._cmap.values():
                if c not in data:
                    raise KeyError(
                        f"States '{self.name}': Coordinate '{c}' not found in dataset {fpath.name}."
                    )
            if set(data.sizes) != set(self._cmap.values()):
                raise ValueError(
                    f"States '{self.name}': Dataset {fpath.name} has unexpected dimensions {data.sizes}, expected {set(self._cmap.values())}."
                )

            # reorder dimensions:
            data = data.transpose(*self._cmap.values())

            # reduce dimensions:
            if algo is not None:
                DatasetStates.preproc_first(
                    cast(DatasetStates, self),
                    algo,
                    data=data,
                    bounds_extra_space=self.bounds_extra_space,
                    height_bounds=self.height_bounds,
                    loaded_data=loaded_data,
                    verbosity=verbosity,
                )
            if self.isel is not None and len(self.isel):
                isel = {c: s for c, s in self.isel.items() if c in data.sizes}
                data = data.isel(**isel)
            if self.sel is not None and len(self.sel):
                sel = {c: s for c, s in self.sel.items() if c in data.sizes}
                data = data.sel(**sel)

            # rename:
            data = data.rename({ncv: v for v, ncv in {**vrs, **self._cmap}.items()})

            # store data:
            self.VARS = self.var("vrs")
            loaded_data["extra_data"][self.VARS] = list(vrs.keys())
            loaded_data["extra_data"][self.DATA] = data

            if verbosity > 1:
                print(f"\nStates '{self.name}': Data loaded")
                print(data)
                print()

    def size(self) -> int:
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return 1

    def index(self) -> list[int]:
        """
        The index list

        Returns
        -------
        indices: array_like
            The index labels of states, or None for default integers

        """
        return [0]

    def calculate(  # type: ignore[override]
        self, algo: Algorithm, mdata: MData, fdata: FData, tdata: TData
    ) -> dict[str, np.ndarray]:
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data

        Returns
        -------
        results: dict[str, numpy.ndarray]
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)

        """
        # prepare
        super().calculate(algo, mdata, fdata, tdata)
        vrs = mdata.extra_data[self.VARS]
        data = mdata.extra_data[self.DATA]
        n_targets = tdata.n_targets
        n_tpoints = tdata.n_tpoints
        points = tdata[FC.TARGETS][0, ...].reshape(n_targets * n_tpoints, 3)

        # get interpolation points:
        pts = {}
        for i, c in enumerate([FV.X, FV.Y, FV.H]):
            if c in self._cmap:
                pts[c] = points[:, i]

        valid = np.ones(n_targets * n_tpoints, dtype=bool)
        for c in self._cmap:
            valid &= np.isfinite(pts[c])

        out = {
            v: np.full(n_targets * n_tpoints, np.nan, dtype=config.dtype_double)
            for v in vrs
        }

        # interpolate through Dataset.interp():
        if np.any(valid):
            pvalid = DataFrame({c: pts[c][valid] for c in self._cmap}).to_xarray()
            results = data.interp(
                **{c: pvalid[c] for c in self._cmap.keys()},
                **self.interp_pars,
            )
            del pvalid

            for v in vrs:
                out[v][valid] = results[v].to_numpy()

        # set interpolated values:
        for v in vrs:
            tdata[v] = out[v].reshape(1, n_targets, n_tpoints)

        # set fixed values:
        for v, d in self.fixed_vars.items():
            tdata[v][:] = d

        # set weights:
        tdata[FV.WEIGHT] = np.ones((1, 1, 1), dtype=config.dtype_double)
        tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)

        return {v: tdata[v] for v in self.output_point_vars(algo)}
