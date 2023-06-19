import numpy as np
from scipy.interpolate import interpn
from tqdm import tqdm

from foxes.core import WakeFrame
from foxes.utils import wd2uv
from foxes.core.data import Data
import foxes.variables as FV
import foxes.constants as FC


class Timelines(WakeFrame):
    """
    Streamline following wakes for timeseries based data

    Attributes
    ----------
    cl_ipars: dict
        Interpolation parameters for centre line
        point interpolation

    :group: models.wake_frames

    """

    def __init__(self, cl_ipars={}):
        """
        Constructor.

        Parameters
        ----------
        cl_ipars: dict
            Interpolation parameters for centre line
            point interpolation

        """
        super().__init__()
        self.cl_ipars = cl_ipars

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        This includes loading all required data from files. The model
        should return all array type data as part of the idata return
        dictionary (and not store it under self, for memory reasons). This
        data will then be chunked and provided as part of the mdata object
        during calculations.

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
        idata = super().initialize(algo, verbosity)
        
        if verbosity > 0:
            print(f"{self.name}: Pre-calculating ambient wind vectors")
        
        # get and check times:
        times = np.asarray(algo.states.index())
        if not np.issubdtype(times.dtype, np.datetime64):
            raise TypeError(f"{self.name}: Expecting state index of type np.datetime64, found {times.dtype}")

        # calculate horizontal wind vector in all states:
        self._uv = np.zeros((algo.n_states, 1, 3), dtype=FC.DTYPE)
            
        # prepare mdata:
        mdata = algo.idata_mem[algo.states.name]["data_vars"]
        mdict = {v: d[1] for v, d in mdata.items()}
        mdims = {v: d[0] for v, d in mdata.items()}
        mdata = Data(mdict, mdims, loop_dims=[FC.STATE])
        del mdict, mdims
        
        # prepare fdata:
        fdata = Data({}, {}, loop_dims=[FC.STATE])
        
        # prepare pdata:
        pdata = {v: np.zeros((algo.n_states, 1), dtype=FC.DTYPE) 
                 for v in algo.states.output_point_vars(algo)}
        pdata[FC.POINTS] = np.zeros((algo.n_states, 1, 3), dtype=FC.DTYPE)
        pdims = {FC.POINTS: (FC.STATE, FC.POINT, FV.XYH)}
        pdims.update({v: (FC.STATE, FC.POINT) for v in pdata.keys()})
        pdata = Data(pdata, pdims, loop_dims=[FC.STATE, FC.POINT])
        
        # calculate:      
        res = algo.states.calculate(algo, mdata, fdata, pdata)
        dt = ( times[1:] - times[:-1] ).astype('timedelta64[s]').astype(FC.ITYPE)
        self._dxy = wd2uv(res[FV.WD], res[FV.WS])[:-1, 0, :2] * dt[:, None]
        self._dxy = np.insert(self._dxy, 0, self._dxy[0], axis=0)
        
        return idata

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
        # n_states, n_turbines, n_points, n_spts
        dists = np.linalg.norm(
            points[:, None, :, None] - spts[:, :, None, :n_spts], axis=-1
        )
        if tcase:
            for ti in range(n_turbines):
                dists[:, ti, ti] = 1e20
        inds = np.argmin(dists, axis=3)
        dists = np.take_along_axis(dists, inds[:, :, :, None], axis=3)[..., 0]
        done = inds < n_spts - 1

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
        del data, spts, sn

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

        # DUMMY, TODO:
        out = np.zeros((mdata.n_states, algo.n_turbines), dtype=FC.ITYPE)
        out[:] = np.arange(algo.n_turbines)[None, :]
        return out

        # prepare:
        n_states = mdata.n_states
        n_turbines = mdata.n_turbines
        sxyh = mdata[self.DATA]
        
        print("HERE")
        print(sxyh)
        print(sxyh.shape)
        quit()
        
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
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        points: numpy.ndarray
            The evaluation points, shape: (n_states, n_points, 3)

        Returns
        -------
        wake_coos: numpy.ndarray
            The wake coordinates, shape: (n_states, n_points, 3)

        """

        # prepare:
        n_states = mdata.n_states
        n_points = points.shape[1]
        stsel = (np.arange(n_states), states_source_turbine)
        rxyz = fdata[FV.TXYH][stsel]
        raxis = wd2uv(fdata[FV.WD][stsel])
        D = fdata[FV.D][stsel]

        wcoos = np.full((n_states, n_points, 3), np.nan, dtype=FC.DTYPE)
        wcoos[:, :, 2] = points[:, :, 2] - rxyz[:, None, 2]
        wcoosx = wcoos[:, :, 0]

        i0 = np.argwhere(algo.states.index() == mdata[FC.STATE][0])[0][0]
        i1 = i0 + mdata.n_states
        dxy = self._dxy#[:i1]

        trace = np.zeros((n_states, n_points, 3), dtype=FC.DTYPE) # x, y, length
        trace[:, :, :2] = points[:, :, :2] - rxyz[:, None, :2]
        done = np.zeros((n_states, n_points), dtype=bool)
        del rxyz

        steps = 0
        while not np.all(done):

            trace0 = trace.copy()

            j0 = max(i0-steps, 0)
            j1 = max(i1-steps, j0)
            n = j1 - j0
            if n < 0:
                break

            si = np.s_[-n:]
            sj = np.s_[j0:j1]
            delta = dxy[sj][:, None, :]
            trace[si][:, :, :2] -= delta
            trace[si][:, :, 2] += np.linalg.norm(delta, axis=-1)

            steps += 1

        print("HERE", steps)
        quit()



        
        print("HELLP", trace.shape,dxy.shape)
        quit()


        sdist = np.zeros(i0, dtype=FC.DTYPE)
        for i in range(i0):
            sdist[i] = np.linalg.norm(np.sum(self._dxy[i:i0], axis=0))
        sel = sdist < 1e4
        sdist = sdist[sel]
        print(self._dxy)
        quit()
        print(s, sdist.shape, sdist)
        quit()
        
        
        return wcoos


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
