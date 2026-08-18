from __future__ import annotations

from abc import abstractmethod
import numpy as np
from scipy.interpolate import interpn
from typing import TYPE_CHECKING, Any, cast

from foxes.utils import new_instance
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

from .data import TData
from .model import Model

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData


class WakeFrame(Model):
    """
    Abstract base class for wake frames.

    Wake frames translate global coordinates into
    wake frame coordinates, which are then evaluated
    by wake models.

    They are also responsible for the calculation of
    the turbine evaluation order.


    """

    @abstractmethod
    def calc_order(self, algo: Algorithm, mdata: MData, fdata: FData) -> np.ndarray:
        """
        Calculates the order of turbine evaluation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data

        Returns
        -------
        order
            The turbine order, shape: (n_states, n_turbines)

        """
        pass

    @abstractmethod
    def get_wake_coos(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
    ) -> np.ndarray:
        """
        Calculate wake coordinates of rotor points.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        downwind_index
            The index of the wake causing turbine
            in the downwind order

        Returns
        -------
        wake_coos
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        """
        pass

    def get_wake_modelling_data(
        self,
        algo: Algorithm,
        variable: str,
        downwind_index: int,
        fdata: FData,
        tdata: TData,
        target: str,
        states0: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return data that is required for computing the
        wake from source turbines to evaluation points.

        Parameters
        ----------
        algo
            The algorithm, needed for data from previous iteration
        variable
            The variable, serves as data key
        downwind_index
            The index in the downwind order
        fdata
            The farm data
        tdata
            The target point data
        target
            The dimensions identifier for the output,
            FC.STATE_TURBINE, FC.STATE_TARGET,
            FC.STATE_TARGET_TPOINT
        states0
            The states of wake creation

        Returns
        -------
        data
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

    def get_centreline_points(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        downwind_index: int,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Gets the points along the centreline for given
        values of x.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        downwind_index
            The index in the downwind order
        x
            The wake frame x coordinates, shape: (n_states, n_points)

        Returns
        -------
        points
            The centreline points, shape: (n_states, n_points, 3)

        """
        raise NotImplementedError(
            f"Wake frame '{self.name}': Centreline points requested but not implemented."
        )

    def calc_centreline_integral(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        downwind_index: int,
        variables: list[str],
        x: np.ndarray,
        dx: float,
        wake_models: list[Any] | None = None,
        self_wake: bool = True,
        **ipars: Any,
    ) -> np.ndarray:
        """
        Integrates variables along the centreline.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        downwind_index
            The index in the downwind order
        variables
            The variables to be integrated
        x
            The wake frame x coordinates of the upper integral bounds,
            shape: (n_states, n_points)
        dx
            The step size of the integral
        wake_models
            The wake models to consider, default: from algo
        self_wake
            Flag for considering only wake from states_source_turbine
        ipars
            Additional interpolation parameters

        Returns
        -------
        results
            The integration results, shape: (n_states, n_points, n_vars)

        """
        # prepare:
        n_states, n_points = x.shape
        vrs = [FV.amb2var.get(v, v) for v in variables]
        n_vars = len(vrs)

        # calc evaluation points:
        xmin = 0.0
        max_wake_length_km = algo.max_wake_length_km
        xmax = min(np.nanmax(x), max_wake_length_km * 1e3)
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
        states = algo.states
        res = states.calculate(algo, mdata, fdata, tdata)
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
    def new(cls, wframe_type: str, *args: Any, **kwargs: Any) -> WakeFrame:
        """
        Run-time wake frame factory.

        Parameters
        ----------
        wframe_type
            The selected derived class name
        args
            Additional parameters for constructor
        kwargs
            Additional parameters for constructor

        """
        return cast(WakeFrame, new_instance(cls, wframe_type, *args, **kwargs))
