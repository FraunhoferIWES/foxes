from abc import abstractmethod
import numpy as np
from scipy.interpolate import interpn

from .data import Data
from .model import Model
import foxes.constants as FC
import foxes.variables as FV


class WakeFrame(Model):
    """
    Abstract base class for wake frames.

    Wake frames translate global coordinates into
    wake frame coordinates, which are then evaluated
    by wake models.

    They are also responsible for the calculation of
    the turbine evaluation order.

    :group: core

    """

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def get_wake_modelling_data(
        self,
        algo,
        variable,
        states_source_turbine,
        fdata,
        pdata,
        states0=None,
    ):
        """
        Return data that is required for computing the
        wake from source turbines to evaluation points.

        Parameters
        ----------
        algo: foxes.core.Algorithm, optional
            The algorithm, needed for data from previous iteration
        variable: str
            The variable, serves as data key
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data
        states0: numpy.ndarray, optional
            The states of wake creation

        """
        n_states = fdata.n_states
        n_points = pdata.n_points
        s = np.arange(n_states) if states0 is None else states0

        out = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        out[:] = fdata[variable][s, states_source_turbine][:, None]

        return out

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
        raise NotImplementedError(
            f"Wake frame '{self.name}': Centreline points requested but not implemented."
        )

    def calc_centreline_integral(
        self,
        algo,
        mdata,
        fdata,
        states_source_turbine,
        variables,
        x,
        dx,
        wake_models=None,
        self_wake=True,
        **ipars,
    ):
        """
        Integrates variables along the centreline.

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
        variables: list of str
            The variables to be integrated
        x: numpy.ndarray
            The wake frame x coordinates of the upper integral bounds,
            shape: (n_states, n_points)
        dx: float
            The step size of the integral
        wake_models: list of foxes.core.WakeModels
            The wake models to consider, default: from algo
        self_wake: bool
            Flag for considering only wake from states_source_turbine
        ipars: dict, optional
            Additional interpolation parameters

        Returns
        -------
        results: numpy.ndarray
            The integration results, shape: (n_states, n_points, n_vars)

        """
        # prepare:
        n_states, n_points = x.shape
        vrs = [FV.amb2var.get(v, v) for v in variables]
        n_vars = len(vrs)

        # calc evaluation points:
        xmin = 0.0
        xmax = np.nanmax(x)
        n_steps = int((xmax - xmin) / dx)
        if xmin + n_steps * dx < xmax:
            n_steps += 1
        n_ix = n_steps + 1
        xs = np.arange(xmin, xmin + n_ix * dx, dx)
        xpts = np.zeros((n_states, n_steps), dtype=FC.DTYPE)
        xpts[:] = xs[None, 1:]
        pts = self.get_centreline_points(
            algo, mdata, fdata, states_source_turbine, xpts
        )

        # run ambient calculation:
        pdata = Data.from_points(
            pts,
            data={v: np.full((n_states, n_steps), np.nan, dtype=FC.DTYPE) for v in vrs},
            dims={v: (FC.STATE, FC.POINT) for v in vrs},
        )
        res = algo.states.calculate(algo, mdata, fdata, pdata)
        pdata.update(res)
        amb2var = algo.get_model("SetAmbPointResults")()
        amb2var.initialize(algo, verbosity=0)
        res = amb2var.calculate(algo, mdata, fdata, pdata)
        pdata.update(res)
        del res, amb2var

        # find out if all vars ambient:
        ambient = True
        for v in variables:
            if v not in FV.amb2var:
                ambient = False
                break

        # calc wakes:
        if not ambient:
            wcalc = algo.get_model("PointWakesCalculation")(
                vrs, wake_models=wake_models
            )
            wcalc.initialize(algo, verbosity=0)
            wsrc = states_source_turbine if self_wake else None
            res = wcalc.calculate(algo, mdata, fdata, pdata, states_source_turbine=wsrc)
            pdata.update(res)
            del wcalc, res

        # collect integration results:
        iresults = np.zeros((n_states, n_ix, n_vars), dtype=FC.DTYPE)
        for vi, v in enumerate(variables):
            for i in range(n_steps):
                iresults[:, i + 1, vi] = iresults[:, i, vi] + pdata[v][:, i] * dx

        # interpolate to x of interest:
        qts = np.zeros((n_states, n_points, 2), dtype=FC.DTYPE)
        qts[:, :, 0] = np.arange(n_states)[:, None]
        qts[:, :, 1] = x
        qts = qts.reshape(n_states * n_points, 2)
        results = interpn(
            (np.arange(n_states), xs),
            iresults,
            qts,
            bounds_error=False,
            fill_value=0.0,
            **ipars,
        )

        return results.reshape(n_states, n_points, n_vars)
