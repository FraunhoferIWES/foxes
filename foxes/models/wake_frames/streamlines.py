import numpy as np
from scipy.interpolate import interpn

from foxes.core import WakeFrame
from foxes.utils import wd2uv
from foxes.core.data import Data
import foxes.variables as FV
import foxes.constants as FC


class Streamlines2D(WakeFrame):
    """
    Streamline following wakes

    Attributes
    ----------
    step: float
        The streamline step size in m
    max_length: float
        The maximal streamline length
    cl_ipars: dict
        Interpolation parameters for centre line
        point interpolation

    :group: models.wake_frames

    """

    def __init__(self, step, max_length=1e4, cl_ipars={}):
        """
        Constructor.

        Parameters
        ----------
        step: float
            The streamline step size in m
        max_length: float
            The maximal streamline length
        cl_ipars: dict
            Interpolation parameters for centre line
            point interpolation

        """
        super().__init__()
        self.step = step
        self.max_length = max_length
        self.cl_ipars = cl_ipars

        self.DATA = self.var("DATA")

    def __repr__(self):
        return super().__repr__() + f"(step={self.step})"
    
    def _calc_streamlines(self, algo, mdata, fdata):
        """
        Helper function that computes all streamline data
        """
        # prepare:
        n_states = mdata.n_states
        n_turbines = mdata.n_turbines
        N = int(self.max_length/self.step)

        # calc data: x, y, z, wd
        data = np.zeros((n_states, n_turbines, N, 4), dtype=FC.DTYPE)
        for i in range(N):

            # set streamline start point data (rotor centre):
            if i == 0:
                data[:, :, i, :3] = fdata[FV.TXYH]
                data[:, :, i, 3] = fdata[FV.AMB_WD]
            
            # compute next step:
            else:

                # calculate next point:
                xyz = data[:, :, i-1, :3]
                n = wd2uv(data[:, :, i-1, 3])
                data[:, :, i, :2] = xyz[:, :, :2] + self.step * n
                data[:, :, i, 2] = xyz[:, :, 2]
                
                # calculate next tangential vector:
                svars = algo.states.output_point_vars(algo)
                pdata = {FC.POINTS: data[:, :, i, :3]}
                pdims = {FC.POINTS: (FC.STATE, FC.POINT, FC.XYH)}
                pdata.update(
                    {v: np.full((n_states, n_turbines), np.nan, dtype=FC.DTYPE) 
                     for v in svars}
                )
                pdims.update({v: (FC.STATE, FC.POINT) for v in svars})
                pdata = Data(pdata, pdims, loop_dims=[FC.STATE, FC.POINT])
                data[:, :, i, 3] = algo.states.calculate(
                                        algo, mdata, fdata, pdata)[FV.WD]
                
                sel = np.isnan(data[:, :, i, 3])
                if np.any(sel):
                    data[sel, i, 3] = data[sel, i-1, 3]

        return data

    def get_streamline_data(self, algo, mdata, fdata):
        """
        Gets streamline data, generating it on the fly
        
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
        data: numpy.ndarray
            The streamline data, shape: 
            (n_states, n_turbines, n_steps, 4)
            with data x, y, z, wd
        
        """
        if (
            self.DATA not in mdata or
            not np.all(mdata[self.DATA][:, :, 0, :3] == fdata[FV.TXYH])
        ):
            mdata[self.DATA] = self._calc_streamlines(algo, mdata, fdata)
        
        return mdata[self.DATA]

    def _calc_coos(self, algo, mdata, fdata, points, states_source_turbine):
        """
        Helper function, calculates streamline coordinates
        for given points and given turbine
        """
        
        # prepare:
        n_states = mdata.n_states
        n_points = points.shape[1]
        st_sel = (np.arange(n_states), states_source_turbine)
        
        # find nearest streamline points:
        data = self.get_streamline_data(algo, mdata, fdata)[st_sel]
        dists = np.linalg.norm(points[:, :, None, :2] - data[:, None, :, :2], axis=-1)
        selp = np.argmin(dists, axis=2)
        data = np.take_along_axis(data[:, None], selp[:, :, None, None], axis=2)[:, :, 0]
        slen = self.step * selp
        del dists, selp

        # calculate coordinates:
        coos = np.zeros((n_states, n_points, 3), dtype=FC.DTYPE)
        nx = wd2uv(data[:, :, 3])
        ny = np.stack([-nx[:, :, 1], nx[:, :, 0]], axis=2)
        delta = points[:, :, :2] - data[:, :, :2]
        coos[:, :, 0] = slen + np.einsum("spd,spd->sp", delta, nx)
        coos[:, :, 1] = np.einsum("spd,spd->sp", delta, ny)
        coos[:, :, 2] = points[:, :, 2] - data[:, :, 2]

        return coos


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
        n_states = fdata.n_states
        n_turbines = algo.n_turbines
        pdata = Data.from_points(points=fdata[FV.TXYH])

        # calculate streamline x coordinates for turbines rotor centre points:
        # n_states, n_turbines_source, n_turbines_target
        coosx = np.zeros((n_states, n_turbines, n_turbines), dtype=FC.DTYPE)
        for ti in range(n_turbines):
            coosx[:, ti, :] = self.get_wake_coos(
                algo, mdata, fdata, pdata, np.full(n_states, ti)
            )[..., 0]

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
        return self._calc_coos(algo, mdata, fdata, 
                               pdata[FC.POINTS], states_source_turbine)

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
        st_sel = (np.arange(n_states), states_source_turbine)
        data = self.get_streamline_data(algo, mdata, fdata)[st_sel]
        spts = data[:, :, :3]
        n_spts = spts.shape[1]
        xs = self.step * np.arange(n_spts)

        # interpolate to x of interest:
        qts = np.zeros((n_states, n_points, 2), dtype=FC.DTYPE)
        qts[:, :, 0] = np.arange(n_states)[:, None]
        qts[:, :, 1] = np.minimum(x, xs[-1])
        qts = qts.reshape(n_states * n_points, 2)
        ipars = dict(bounds_error=False, fill_value=0.0)
        ipars.update(self.cl_ipars)
        results = interpn((np.arange(n_states), xs), spts, qts, **ipars)

        return results.reshape(n_states, n_points, 3)
    