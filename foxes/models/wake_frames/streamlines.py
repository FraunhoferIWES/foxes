import numpy as np
from scipy.interpolate import interpn

from foxes.core import WakeFrame
from foxes.utils import wd2uv
from foxes.core.data import Data
import foxes.variables as FV
import foxes.constants as FC


class Streamlines(WakeFrame):
    """
    Streamline following wakes

    Attributes
    ----------
    step: float
        The streamline step size in m
    n_delstor: int
        The streamline point storage increase
    max_length: float
        The maximal streamline length
    cl_ipars: dict
        Interpolation parameters for centre line
        point interpolation

    :group: models.wake_frames

    """

    def __init__(self, step, n_delstor=100, max_length=1e5, cl_ipars={}):
        """
        Constructor.

        Parameters
        ----------
        step: float
            The streamline step size in m
        n_delstor: int
            The streamline point storage increase
        max_length: float
            The maximal streamline length
        cl_ipars: dict
            Interpolation parameters for centre line
            point interpolation

        """
        super().__init__()
        self.step = step
        self.n_delstor = n_delstor
        self.max_length = max_length
        self.cl_ipars = cl_ipars

    def __repr__(self):
        return super().__repr__() + f"(step={self.step})"

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().initialize(algo, verbosity)
        self.DATA = self.var("DATA")
        self.CNTR = self.var("CNTR")
        self.PRES = self.var("PRES")

    def _init_data(self, mdata, fdata):
        # prepare:
        n_states = mdata.n_states
        n_turbines = mdata.n_turbines

        # x, y, z, u, v, w, len
        mdata[self.DATA] = np.full(
            (n_states, n_turbines, self.n_delstor, 7), np.nan, dtype=FC.DTYPE
        )
        mdata[self.CNTR] = 1

        # set streamline start point data (rotor centre):
        mdata[self.DATA][:, :, 0, :3] = fdata[FV.TXYH]
        mdata[self.DATA][:, :, 0, 3:5] = wd2uv(fdata[FV.AMB_WD])
        mdata[self.DATA][:, :, 0, 5:] = 0.0

    def _add_next_point(self, algo, mdata, fdata):
        """
        Helper function, adds next point to streamlines.
        """

        # prepare:
        n_states = mdata.n_states
        n_turbines = mdata.n_turbines
        n_spts = int(mdata[self.CNTR])
        data = mdata[self.DATA]

        # ensure storage size:
        while n_spts >= data.shape[2]:
            data = np.append(
                data,
                np.full(
                    (n_states, n_turbines, self.n_delstor, 7),
                    np.nan,
                    dtype=FC.DTYPE,
                ),
                axis=2,
            )
            mdata[self.DATA] = data

        # data aliases:
        spts = data[..., :3]
        sn = data[..., 3:6]
        slen = data[..., 6]

        # calculate next point:
        p0 = spts[:, :, n_spts - 1]
        n0 = sn[:, :, n_spts - 1]
        spts[:, :, n_spts] = p0 + self.step * n0
        slen[:, :, n_spts] = slen[:, :, n_spts - 1] + self.step
        newpts = spts[:, :, n_spts]
        del p0, n0

        # calculate next tangential vector:
        svars = algo.states.output_point_vars(algo)
        pdata = {FC.POINTS: newpts}
        pdims = {FC.POINTS: (FC.STATE, FC.POINT, FC.XYH)}
        pdata.update(
            {v: np.full((n_states, n_turbines), np.nan, dtype=FC.DTYPE) for v in svars}
        )
        pdims.update({v: (FC.STATE, FC.POINT) for v in svars})
        pdata = Data(pdata, pdims, loop_dims=[FC.STATE, FC.POINT])
        data[:, :, n_spts, 5] = 0.0
        data[:, :, n_spts, 3:5] = wd2uv(
            algo.states.calculate(algo, mdata, fdata, pdata)[FV.WD]
        )
        mdata[self.CNTR] += 1

        return newpts, data, mdata[self.CNTR]

    def _calc_coos(self, algo, mdata, fdata, points, tcase=False):
        """
        Helper function, calculates streamline coordinates
        for given points.
        """

        # prepare:
        n_states = mdata.n_states
        n_turbines = mdata.n_turbines
        n_points = points.shape[1]
        n_spts = int(mdata[self.CNTR])
        data = mdata[self.DATA]
        spts = data[..., :3]
        sn = data[..., 3:6]

        # find minimal distances to existing streamline points:
        # (loop over target points, since otherwise this blows memory)
        done = np.zeros((n_states, n_turbines, n_points), dtype=bool)
        inds = np.full((n_states, n_turbines, n_points), -1, dtype=FC.ITYPE)
        dists = np.full((n_states, n_turbines, n_points), np.nan, dtype=FC.DTYPE)
        for pi in range(n_points):
            hdists = np.linalg.norm(
                points[:, None, pi, None] - spts[:, :, :n_spts], axis=-1
            )
            if tcase:
                hdists[:, pi] = np.inf
            inds[:, :, pi] = np.argmin(hdists, axis=2)
            dists[:, :, pi] = np.take_along_axis(hdists, inds[:, :, pi, None], axis=2)[
                ..., 0
            ]
            done[:, :, pi] = inds[:, :, pi] < n_spts - 1
            del hdists

        # calc streamline points, as many as needed:
        maxl = np.nanmax(data[:, :, n_spts - 1, 6])
        while maxl + self.step <= self.max_length and not np.all(done):
            # print("CALC STREAMLINES, TODO", np.sum(~done))

            # add next streamline point:
            newpts, data, n_spts = self._add_next_point(algo, mdata, fdata)

            # evaluate distance:
            d = np.linalg.norm(points[:, None] - newpts[:, :, None], axis=-1)
            if tcase:
                for ti in range(n_turbines):
                    d[:, ti, ti] = 1e20
            sel = d < dists
            if np.any(sel):
                dists[sel] = d[sel]
                inds[sel] = n_spts - 1

            # rotation:
            done = inds < n_spts - 1
            maxl = np.nanmax(data[:, :, n_spts - 1, 6])
            del newpts

        # shrink to size:
        mdata[self.DATA] = data[:, :, :n_spts]
        del data, spts, sn, dists, done

        # select streamline points:
        # n_states, n_turbines, n_points, 7
        data = np.take_along_axis(
            mdata[self.DATA][:, :, :, None], inds[:, :, None, :, None], axis=2
        )[:, :, 0]
        spts = data[..., :3]
        sn = data[..., 3:6]
        slen = data[..., 6]

        # calculate coordinates:
        coos = np.zeros((n_states, n_turbines, n_points, 3), dtype=FC.DTYPE)
        delta = points[:, None] - spts
        nx = sn
        nz = np.array([0.0, 0.0, 1.0], dtype=FC.DTYPE)[None, None, None, :]
        ny = np.cross(nz, nx, axis=-1)
        coos[..., 0] = slen + np.einsum("stpd,stpd->stp", delta, nx)
        coos[..., 1] = np.einsum("stpd,stpd->stp", delta, ny)
        coos[..., 2] = delta[..., 2]

        return coos

    def _ensure_min_length(self, algo, mdata, fdata, length):
        """
        Helper function, ensures minimal length of streamlines
        """
        data = mdata[self.DATA]
        slen = data[:, :, mdata[self.CNTR] - 1, 6]
        minl = np.nanmin(slen)
        maxl = np.nanmax(slen)
        while maxl + self.step <= self.max_length and minl < length:
            __, data, n_spts = self._add_next_point(algo, mdata, fdata)
            slen = data[:, :, n_spts - 1, 6]
            minl = np.nanmin(slen)
            maxl = np.nanmax(slen)

    def calc_order(self, algo, mdata, fdata):
        """ "
        Calculates the order of turbine evaluation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data

        Returns
        -------
        order: numpy.ndarray
            The turbine order, shape: (n_states, n_turbines)

        """

        # prepare:
        n_states = mdata.n_states
        n_turbines = mdata.n_turbines

        # initialize storage:
        if self.DATA not in mdata:
            self._init_data(mdata, fdata)

        # calculate streamline x coordinates for turbines rotor centre points:
        # n_states, n_turbines_source, n_turbines_target
        coosx = self._calc_coos(algo, mdata, fdata, fdata[FV.TXYH], tcase=True)[..., 0]

        # derive turbine order:
        # TODO: Remove loop over states
        order = np.zeros((n_states, n_turbines), dtype=FC.ITYPE)
        for si in range(n_states):
            order[si] = np.lexsort(keys=coosx[si])

        return order

    def get_wake_coos(self, algo, mdata, fdata, pdata, states_source_turbine):
        """
        Calculate wake coordinates.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)

        Returns
        -------
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_points, 3)

        """

        # prepare:
        n_states = mdata.n_states
        stsel = (np.arange(n_states), states_source_turbine)
        points = pdata[FC.POINTS]
        pid = id(points)

        # initialize storage:
        if self.DATA not in mdata:
            self._init_data(mdata, fdata)

        # calc streamlines, once for given points:
        if self.PRES not in mdata or pid not in mdata[self.PRES]:
            mdata[self.PRES] = {
                pid: self._calc_coos(algo, mdata, fdata, points, tcase=False)
            }

        return mdata[self.PRES][pid][stsel]

    def get_centreline_points(self, algo, mdata, fdata, states_source_turbine, x):
        """
        Gets the points along the centreline for given
        values of x.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        x: numpy.ndarray
            The wake frame x coordinates, shape: (n_states, n_points)

        Returns
        -------
        points: numpy.ndarray
            The centreline points, shape: (n_states, n_points, 3)

        """
        # calculate long enough streamlines:
        xmax = np.max(x)
        self._ensure_min_length(algo, mdata, fdata, xmax)

        # get streamline points:
        n_states, n_points = x.shape
        data = mdata[self.DATA][range(n_states), states_source_turbine]
        spts = data[:, :, :3]
        n_spts = spts.shape[1]
        xs = self.step * np.arange(n_spts)

        # interpolate to x of interest:
        qts = np.zeros((n_states, n_points, 2), dtype=FC.DTYPE)
        qts[:, :, 0] = np.arange(n_states)[:, None]
        qts[:, :, 1] = x
        qts = qts.reshape(n_states * n_points, 2)
        ipars = dict(bounds_error=False, fill_value=0.0)
        ipars.update(self.cl_ipars)
        results = interpn((np.arange(n_states), xs), spts, qts, **ipars)

        return results.reshape(n_states, n_points, 3)
