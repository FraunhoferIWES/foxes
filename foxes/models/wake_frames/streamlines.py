import numpy as np

from foxes.core import WakeFrame
from foxes.utils import wd2uv
from foxes.core.data import Data
import foxes.variables as FV
import foxes.constants as FC


class Streamlines(WakeFrame):
    """
    Streamline following wakes

    Parameters
    ----------
    step : float
        The streamline step size in m
    n_delstor : int
        The streamline point storage increase

    Attributes
    ----------
    step : float
        The streamline step size in m
    n_delstor : int
        The streamline point storage increase

    """

    def __init__(self, step, n_delstor=100):
        super().__init__()
        self.step = step
        self.n_delstor = n_delstor

    def __repr__(self):
        return super().__repr__() + f"(step={self.step})"

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level

        """
        self.DATA = self.var("DATA")
        self.CNTR = self.var("CNTR")
        self.PRES = self.var("PRES")
        super().initialize(algo, verbosity)

    def _calc_coos(self, algo, mdata, fdata, points, tcase=False):
        """
        Helper function, calculates streamline coordinates
        for given points.
        """

        # prepare:
        n_states = mdata.n_states
        n_turbines = mdata.n_turbines
        n_points = points.shape[1]
        n_spts = mdata[self.CNTR]
        data = mdata[self.DATA]
        spts = data[..., :3]
        sn = data[..., 3:6]
        slen = data[..., 6]

        # find minimal distances to existing streamline points:
        # n_states, n_turbines, n_points, n_spts
        dists = np.linalg.norm(
            points[:, None, :, None] - spts[:, :, None, :n_spts], axis=-1
        )
        if tcase:
            for ti in range(n_turbines):
                dists[:, ti, ti] = 1e20
        inds = np.argmin(dists, axis=3)
        dists = np.take_along_axis(dists, inds[:, :, :, None], axis=3)[..., 0]

        # add streamline points, as many as needed:
        done = inds < n_spts - 1
        while not np.all(done):

            # print("CALC STREAMLINES, TODO", np.sum(~done))

            # ensure storage size:
            if n_spts == data.shape[2]:
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
            pdata = {FV.POINTS: newpts}
            pdims = {FV.POINTS: (FV.STATE, FV.POINT, FV.XYH)}
            pdata.update(
                {
                    v: np.full((n_states, n_turbines), np.nan, dtype=FC.DTYPE)
                    for v in svars
                }
            )
            pdims.update({v: (FV.STATE, FV.POINT) for v in svars})
            pdata = Data(pdata, pdims, loop_dims=[FV.STATE, FV.POINT])
            data[:, :, n_spts, 3:5] = wd2uv(
                algo.states.calculate(algo, mdata, fdata, pdata)[FV.WD]
            )
            data[:, :, n_spts, 5] = 0.0
            del pdims, svars, pdata

            # evaluate distance:
            d = np.linalg.norm(points[:, None] - newpts[:, :, None], axis=-1)
            if tcase:
                for ti in range(n_turbines):
                    d[:, ti, ti] = 1e20
            sel = d < dists
            if np.any(sel):
                dists[sel] = d[sel]
                inds[sel] = n_spts

            # rotation:
            mdata[self.CNTR] += 1
            n_spts = mdata[self.CNTR]
            done = inds < n_spts - 1

        # shrink to size:
        mdata[self.DATA] = data[:, :, :n_spts]
        del data, spts, sn, slen

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
        mdata[self.DATA][:, :, 0, 5] = 0.0
        mdata[self.DATA][:, :, 0, 6] = 0.0

    def calc_order(self, algo, mdata, fdata):
        """ "
        Calculates the order of turbine evaluation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data

        Returns
        -------
        order : numpy.ndarray
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

    def get_wake_coos(self, algo, mdata, fdata, states_source_turbine, points):
        """
        Calculate wake coordinates.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        points : numpy.ndarray
            The evaluation points, shape: (n_states, n_points, 3)

        Returns
        -------
        wake_coos : numpy.ndarray
            The wake coordinates, shape: (n_states, n_points, 3)

        """

        # prepare:
        n_states = mdata.n_states
        stsel = (np.arange(n_states), states_source_turbine)
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
