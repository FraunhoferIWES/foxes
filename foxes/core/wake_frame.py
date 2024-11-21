from abc import abstractmethod
import numpy as np
from scipy.interpolate import interpn

from foxes.utils import all_subclasses
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

from .data import TData
from .model import Model


class WakeFrame(Model):
    """
    Abstract base class for wake frames.

    Wake frames translate global coordinates into
    wake frame coordinates, which are then evaluated
    by wake models.

    They are also responsible for the calculation of
    the turbine evaluation order.

    Attributes
    ----------
    max_length_km: float
        The maximal wake length in km

    :group: core

    """

    def __init__(self, max_length_km=3e4):
        """
        Constructor.

        Parameters
        ----------
        max_length_km: float
            The maximal wake length in km

        """
        super().__init__()
        self.max_length_km = max_length_km

    @abstractmethod
    def calc_order(self, algo, mdata, fdata):
        """
        Calculates the order of turbine evaluation.

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

        Returns
        -------
        order: numpy.ndarray
            The turbine order, shape: (n_states, n_turbines)

        """
        pass

    @abstractmethod
    def get_wake_coos(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
    ):
        """
        Calculate wake coordinates of rotor points.

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
        downwind_index: int
            The index of the wake causing turbine
            in the downwind order

        Returns
        -------
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        """
        pass

    def get_wake_modelling_data(
        self,
        algo,
        variable,
        downwind_index,
        fdata,
        tdata,
        target,
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
        downwind_index: int, optional
            The index in the downwind order
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        target: str, optional
            The dimensions identifier for the output,
            FC.STATE_TURBINE, FC.STATE_TARGET,
            FC.STATE_TARGET_TPOINT
        states0: numpy.ndarray, optional
            The states of wake creation

        Returns
        -------
        data: numpy.ndarray
            Data for wake modelling, shape:
            (n_states, n_turbines) or (n_states, n_target)

        """
        s = np.s_[:] if states0 is None else states0

        if target == FC.STATE_TARGET_TPOINT:
            out = fdata[variable][s, downwind_index, None, None]
        elif target in [FC.STATE_TURBINE, FC.STATE_TARGET]:
            out = fdata[variable][s, downwind_index, None]
        else:
            raise ValueError(
                f"Unkown target '{target}', choices are {FC.STATE_TURBINE}, {FC.STATE_TARGET}, {FC.STATE_TARGET_TPOINT}"
            )

        return out

    def get_centreline_points(self, algo, mdata, fdata, downwind_index, x):
        """
        Gets the points along the centreline for given
        values of x.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        downwind_index: int
            The index in the downwind order
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
        downwind_index,
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
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        downwind_index: int
            The index in the downwind order
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
        xmax = min(np.nanmax(x), self.max_length_km * 1e3)
        n_steps = int((xmax - xmin) / dx)
        if xmin + n_steps * dx < xmax:
            n_steps += 1
        n_ix = n_steps + 1
        xs = np.arange(xmin, xmin + n_ix * dx, dx)
        xpts = np.zeros((n_states, n_steps), dtype=config.dtype_double)
        xpts[:] = xs[None, 1:]
        pts = self.get_centreline_points(algo, mdata, fdata, downwind_index, xpts)

        # run ambient calculation:
        tdata = TData.from_points(
            pts,
            data={
                v: np.full((n_states, n_steps, 1), np.nan, dtype=config.dtype_double)
                for v in vrs
            },
            dims={v: (FC.STATE, FC.TARGET, FC.TPOINT) for v in vrs},
        )
        res = algo.states.calculate(algo, mdata, fdata, tdata)
        tdata.update(res)
        amb2var = algo.get_model("SetAmbPointResults")()
        amb2var.initialize(algo, verbosity=0, force=True)
        res = amb2var.calculate(algo, mdata, fdata, tdata)
        tdata.update(res)
        del res, amb2var

        # find out if all vars ambient:
        ambient = True
        for v in variables:
            if v not in FV.amb2var:
                ambient = False
                break

        # calc wakes:
        if not ambient:
            wcalc = algo.get_model("PointWakesCalculation")(wake_models=wake_models)
            wcalc.initialize(algo, verbosity=0, force=True)
            wsrc = downwind_index if self_wake else None
            res = wcalc.calculate(algo, mdata, fdata, tdata, downwind_index=wsrc)
            tdata.update(res)
            del wcalc, res

        # collect integration results:
        iresults = np.zeros((n_states, n_ix, n_vars), dtype=config.dtype_double)
        for vi, v in enumerate(variables):
            for i in range(n_steps):
                iresults[:, i + 1, vi] = iresults[:, i, vi] + tdata[v][:, i, 0] * dx

        # interpolate to x of interest:
        qts = np.zeros((n_states, n_points, 2), dtype=config.dtype_double)
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

    @classmethod
    def new(cls, wframe_type, *args, **kwargs):
        """
        Run-time wake frame factory.

        Parameters
        ----------
        wframe_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """

        if wframe_type is None:
            return None

        allc = all_subclasses(cls)
        found = wframe_type in [scls.__name__ for scls in allc]

        if found:
            for scls in allc:
                if scls.__name__ == wframe_type:
                    return scls(*args, **kwargs)

        else:
            estr = (
                "Wake frame type '{}' is not defined, available types are \n {}".format(
                    wframe_type, sorted([i.__name__ for i in allc])
                )
            )
            raise KeyError(estr)
