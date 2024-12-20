import numpy as np

from foxes.core.states import States
from foxes.config import config, get_input_path
from foxes.utils.wrg_utils import ReaderWRG
from foxes.data import STATES
import foxes.variables as FV
import foxes.constants as FC


class WRGStates(States):
    """
    Ambient states based on WRG data

    Attributes
    ----------
    wrg_fname: str
        Name of the WRG file
    ws_bins: numpy.ndarray
        The wind speed bins, including
        lower and upper bounds, shape: (n_ws_bins+1,)
    fixed_vars: dict
        Fixed uniform variable values, instead of
        reading from data
    bounds_extra_space: float or str
        The extra space, either float in m,
        or str for units of D, e.g. '2.5D'

    :group: input.states

    """

    def __init__(
        self,
        wrg_fname,
        ws_bins,
        fixed_vars={},
        bounds_extra_space="1D",
        **kwargs,
    ):
        """
        Constructor

        Parameters
        ----------
        wrg_fname: str
            Name of the WRG file
        ws_bins: list of float
            The wind speed bins, including
            lower and upper bounds
        fixed_vars: dict
            Fixed uniform variable values, instead of
            reading from data
        bounds_extra_space: float or str, optional
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        kwargs: dict, optional
            Parameters for the base class

        """
        super().__init__(**kwargs)
        self.wrg_fname = wrg_fname
        self.ws_bins = np.asarray(ws_bins)
        self.fixed_vars = fixed_vars
        self.bounds_extra_space = bounds_extra_space

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
        # read wrg file:
        fpath = get_input_path(self.wrg_fname)
        if not fpath.is_file():
            if verbosity > 0:
                print(
                    f"States '{self.name}': Reading static data '{self.wrg_fname}' from context '{STATES}'"
                )
            fpath = algo.dbook.get_file_path(STATES, self.wrg_fname, check_raw=False)
            if verbosity > 0:
                print(f"Path: {fpath}")
        elif verbosity:
            print(f"States '{self.name}': Reading file {fpath}")
        wrg = ReaderWRG(fpath)
        self._p0 = np.array([wrg.x0, wrg.y0], dtype=config.dtype_double)
        self._nx = wrg.nx
        self._ny = wrg.ny
        self._ns = wrg.n_sectors
        self._res = wrg.resolution

        # find bounds:
        if self.bounds_extra_space is not None:
            xy_min, xy_max = algo.farm.get_xy_bounds(
                extra_space=self.bounds_extra_space, algo=algo
            )
            if verbosity > 0:
                print(
                    f"States '{self.name}': Restricting to bounds {xy_min} - {xy_max}"
                )
            xy_min -= self._p0
            xy_max -= self._p0
            ij_min = np.asarray(xy_min / self._res, dtype=config.dtype_int)
            ij_max = np.asarray(xy_max / self._res, dtype=config.dtype_int) + 1
            sx = slice(ij_min[0], ij_max[0])
            sy = slice(ij_min[1], ij_max[1])
        else:
            sx = np.s_[:]
            sy = np.s_[:]

        # store data:
        A = []
        k = []
        fs = []
        for s in range(self._ns):
            A.append(wrg.data[f"A_{s}"].to_numpy().reshape(self._ny, self._nx)[sy, sx])
            k.append(wrg.data[f"k_{s}"].to_numpy().reshape(self._ny, self._nx)[sy, sx])
            fs.append(
                wrg.data[f"fs_{s}"].to_numpy().reshape(self._ny, self._nx)[sy, sx]
            )
        del wrg
        A = np.stack(A, axis=0).T
        k = np.stack(k, axis=0).T
        fs = np.stack(fs, axis=0).T
        self._data = np.stack([A, k, fs], axis=-1)  # (x, y, wd, AKfs)

        # store ws and wd:
        self.WSWD = self.var("WSWD")
        self._wds = np.arange(0.0, 360.0, 360 / self._ns)
        self._wsd = self.ws_bins[1:] - self.ws_bins[:-1]
        self._wss = 0.5 * (self.ws_bins[:-1] + self.ws_bins[1:])
        self._N = len(self._wss) * self._ns
        wswd = np.zeros((len(self._wss), self._ns, 2), dtype=config.dtype_double)
        wswd[..., 0] = self._wss[:, None]
        wswd[..., 1] = self._wds[None, :]
        wswd = wswd.reshape(self._N, 2)
        idata = super().load_data(algo, verbosity)
        idata.coords[self.WSWD] = [self.var(FV.WS), self.var(FV.WD)]
        idata.data_vars[self.WSWD] = ((FC.STATE, self.WSWD), wswd)

        return idata

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self._N

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
        ovars = set([FV.WS, FV.WD, FV.WEIGHT])
        ovars.update(self.fixed_vars.values())
        return list(ovars)
