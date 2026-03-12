import numpy as np
from pandas import DataFrame
from xarray import Dataset, open_dataset

from foxes.core import States
from foxes.config import config, get_input_path
from foxes.data import STATES
import foxes.variables as FV
import foxes.constants as FC

from .dataset_states import DatasetStates


class SingleStateField(States):
    """
    Single state field, i.e. no state coordinate, just a regular grid.

    Attributes
    ----------
    data_source: xarray.Dataset or str
        The NetCDF dataset to read from, or a path to it.
    output_vars: list of str
        List of variable names to read.
    var2ncvar: dict
        Mapping from variable names to netCDF variable names.
    fixed_vars: dict
        Mapping from variable names to fixed values.
    x_coord: str
        Name of the x coordinate.
    y_coord: str
        Name of the y coordinate.
    h_coord: str
        Name of the height coordinate.
    sel: dict
        Subset selection via xr.Dataset.sel()
    isel: dict
        Subset selection via xr.Dataset.isel()
    interp_pars: dict
        Interpolation parameters, passed to the interpolation function.
    bounds_extra_space: float or str
        The extra space, either float in m,
        or str for units of D, e.g. '2.5D'
    height_bounds: tuple
        The (h_min, h_max) height bounds in m. Defaults to H +/- 0.5*D

    :group: input.states

    """

    def __init__(
        self,
        data_source,
        output_vars,
        var2ncvar={},
        fixed_vars={},
        x_coord="x",
        y_coord="y",
        h_coord="height",
        sel=None,
        isel=None,
        interp_pars={},
        bounds_extra_space=1000,
        height_bounds=None,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source: xarray.Dataset or str
            The NetCDF dataset to read from, or a path to it.
        output_vars: list of str
            List of variable names to read.
        var2ncvar: dict
            Mapping from variable names to netCDF variable names.
        fixed_vars: dict
            Mapping from variable names to fixed values.
        x_coord: str
            Name of the x coordinate.
        y_coord: str
            Name of the y coordinate.
        h_coord: str
            Name of the height coordinate.
        sel: dict, optional
            Subset selection via xr.Dataset.sel()
        isel: dict, optional
            Subset selection via xr.Dataset.isel()
        interp_pars: dict
            Interpolation parameters, passed to the interpolation function.
        bounds_extra_space: float or str, optional
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        height_bounds: tuple, optional
            The (h_min, h_max) height bounds in m. Defaults to H +/- 0.5*D
        kwargs: dict
            Keyword arguments passed to the base class.

        """
        super().__init__(**kwargs)
        self.data_source = data_source
        self.output_vars = output_vars
        self.var2ncvar = var2ncvar
        self.fixed_vars = fixed_vars
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.h_coord = h_coord
        self.sel = sel
        self.isel = isel
        self.interp_pars = interp_pars
        self.bounds_extra_space = bounds_extra_space
        self.height_bounds = height_bounds

        self._cmap = {
            FV.X: self.x_coord,
            FV.Y: self.y_coord,
        }
        if self.h_coord is not None:
            self._cmap[FV.H] = self.h_coord

        self._data = None

    def output_point_vars(self, algo):
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
    def data(self):
        """
        The field data

        Returns
        -------
        d: xrarray.Dataset
            The field data

        """
        return self._data

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
        if self._data is None:
            # read NetCDF data file, if not given as Dataset already:
            if isinstance(self.data_source, Dataset):
                self._data = self.data_source
            else:
                fpath = get_input_path(self.data_source)
                if not fpath.is_file():
                    if verbosity > 0:
                        print(
                            f"States '{self.name}': Reading static data '{fpath.name}' from context '{STATES}'"
                        )
                    fpath = algo.dbook.get_file_path(
                        STATES, fpath.name, check_raw=False
                    )
                    if verbosity > 0:
                        print(f"Path: {fpath}")
                elif verbosity:
                    print(f"States '{self.name}': Reading file {fpath}")
                self._data = open_dataset(fpath)

            # remove unnecessary variables:
            self._vars = {
                var: self.var2ncvar.get(var, var)
                for var in self.output_vars
                if var not in self.fixed_vars
            }
            try:
                self._data = self._data[list(self._vars.values())]
            except KeyError as e:
                raise KeyError(
                    f"States '{self.name}': Variable '{e.args[0]}' not found in dataset {fpath.name}."
                )

            # check coordinates:
            for c in self._cmap.values():
                if c not in self._data:
                    raise KeyError(
                        f"States '{self.name}': Coordinate '{c}' not found in dataset {fpath.name}."
                    )
            if set(self._data.sizes) != set(self._cmap.values()):
                raise ValueError(
                    f"States '{self.name}': Dataset {fpath.name} has unexpected dimensions {self._data.sizes}, expected {set(self._cmap.values())}."
                )

            # reorder dimensions:
            self._data = self._data.transpose(*self._cmap.values())

            # reduce dimensions:
            DatasetStates.preproc_first(
                self,
                algo,
                data=self._data,
                cmap=self._cmap,
                vars=None,
                bounds_extra_space=self.bounds_extra_space,
                height_bounds=self.height_bounds,
                verbosity=verbosity,
            )
            if self.isel is not None and len(self.isel):
                isel = {c: s for c, s in self.isel.items() if c in self._data.sizes}
                self._data = self._data.isel(**isel)
            if self.sel is not None and len(self.sel):
                sel = {c: s for c, s in self.sel.items() if c in self._data.sizes}
                self._data = self._data.sel(**sel)

            # rename:
            self._data = self._data.rename(
                {ncv: v for v, ncv in {**self._vars, **self._cmap}.items()}
            )

            if verbosity > 1:
                print(f"\nStates '{self.name}': Data loaded")
                print(self._data)
                print()

        return super().load_data(algo, verbosity=verbosity)

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return 1

    def index(self):
        """
        The index list

        Returns
        -------
        indices: array_like
            The index labels of states, or None for default integers

        """
        return [0]

    def calculate(self, algo, mdata, fdata, tdata):
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
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)

        """
        # prepare
        self.ensure_output_vars(algo, tdata)
        n_targets = tdata.n_targets
        n_tpoints = tdata.n_tpoints
        points = tdata[FC.TARGETS][0, ...].reshape(n_targets * n_tpoints, 3)

        # get interpolation points:
        pts = {}
        for i, c in enumerate([FV.X, FV.Y, FV.H]):
            if c in self._cmap:
                pts[c] = points[:, i]

        # interpolate through Dataset.interp():
        pts = DataFrame(pts).to_xarray()
        results = self.data.interp(
            **{c: pts[c] for c in self._cmap.keys()},
            **self.interp_pars,
        )
        del pts

        # set interpolated values:
        for v in self._vars.keys():
            tdata[v] = results[v].to_numpy().reshape(1, n_targets, n_tpoints)

        # set fixed values:
        for v, d in self.fixed_vars.items():
            tdata[v][:] = d

        # set weights:
        tdata[FV.WEIGHT] = np.ones((1, 1, 1), dtype=config.dtype_double)
        tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)

        return {v: tdata[v] for v in self.output_point_vars(algo)}
