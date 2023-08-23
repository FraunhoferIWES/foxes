import numpy as np
from abc import abstractmethod

import foxes.variables as FV
import foxes.constants as FC
from .farm_data_model import FarmDataModel
from .data import Data
from foxes.utils import wd2uv, uv2wd


class RotorModel(FarmDataModel):
    """
    Abstract base class of rotor models.

    Rotor models calculate ambient farm data from
    states, and provide rotor points and weights
    for the calculation of rotor effective quantities.

    Attributes
    ----------
    calc_vars: list of str
        The variables that are calculated by the model
        (Their ambients are added automatically)

    :group: core

    """

    def __init__(self, calc_vars):
        """
        Constructor.

        Parameters
        ----------
        calc_vars: list of str
            The variables that are calculated by the model
            (Their ambients are added automatically)

        """
        super().__init__()
        self.calc_vars = calc_vars

    def output_farm_vars(self, algo):
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
        return list(
            set(
                self.calc_vars
                + [FV.var2amb[v] for v in self.calc_vars if v in FV.var2amb]
            )
        )

    @abstractmethod
    def n_rotor_points(self):
        """
        The number of rotor points

        Returns
        -------
        n_rpoints: int
            The number of rotor points

        """
        pass

    @abstractmethod
    def rotor_point_weights(self):
        """
        The weights of the rotor points

        Returns
        -------
        weights: numpy.ndarray
            The weights of the rotor points,
            add to one, shape: (n_rpoints,)

        """
        pass

    @abstractmethod
    def design_points(self):
        """
        The rotor model design points.

        Design points are formulated in rotor plane
        (x,y,z)-coordinates in rotor frame, such that
        - (0,0,0) is the centre point,
        - (1,0,0) is the point radius * n_rotor_axis
        - (0,1,0) is the point radius * n_rotor_side
        - (0,0,1) is the point radius * n_rotor_up

        Returns
        -------
        dpoints: numpy.ndarray
            The design points, shape: (n_points, 3)

        """
        pass

    def get_rotor_points(self, algo, mdata, fdata):
        """
        Calculates rotor points from design points.

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
        points: numpy.ndarray
            The rotor points, shape:
            (n_states, n_turbines, n_rpoints, 3)

        """

        n_states = mdata.n_states
        n_points = self.n_rotor_points()
        n_turbines = algo.n_turbines
        dpoints = self.design_points()
        D = fdata[FV.D]

        rax = np.zeros((n_states, n_turbines, 3, 3), dtype=FC.DTYPE)
        n = rax[:, :, 0, 0:2]
        m = rax[:, :, 1, 0:2]
        n[:] = wd2uv(fdata[FV.YAW], axis=-1)
        m[:] = np.stack([-n[:, :, 1], n[:, :, 0]], axis=-1)
        rax[:, :, 2, 2] = 1

        points = np.zeros((n_states, n_turbines, n_points, 3), dtype=FC.DTYPE)
        points[:] = fdata[FV.TXYH][:, :, None, :]
        points[:] += (
            0.5 * D[:, :, None, None] * np.einsum("stad,pa->stpd", rax, dpoints)
        )

        return points

    def _set_res(self, fdata, v, res, stsel):
        """
        Helper function for results setting
        """
        if stsel is None:
            fdata[v] = res.copy()
        elif res.shape[1] == 1:
            fdata[v][stsel] = res[:, 0]
        else:
            raise ValueError(
                f"Rotor model '{self.name}': states_turbine is not None, but results shape for '{v}' has more than one turbine, {res.shape}"
            )

    def eval_rpoint_results(
        self,
        algo,
        mdata,
        fdata,
        rpoint_results,
        weights,
        states_turbine=None,
        copy_to_ambient=False,
    ):
        """
        Evaluate rotor point results.

        This function modifies `fdata`, either
        for all turbines or one turbine per state,
        depending on parameter `states_turbine`. In
        the latter case, the turbine dimension of the
        `rpoint_results` is expected to have size one.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        rpoint_results: dict
            The results at rotor points. Keys: variable str.
            Values: numpy.ndarray, shape if `states_turbine`
            is None: (n_states, n_turbines, n_rpoints).
            Else: (n_states, 1, n_rpoints)
        weights: numpy.ndarray
            The rotor point weights, shape: (n_rpoints,)
        states_turbine: numpy.ndarray of int, optional
            The turbine indices, one per state. Shape: (n_states,)
        copy_to_ambient: bool, optional
            If `True`, the fdata results are copied to ambient
            variables after calculation

        """

        n_states = mdata.n_states
        n_turbines = algo.n_turbines
        if states_turbine is not None:
            stsel = (np.arange(n_states), states_turbine)
        else:
            stsel = None

        uvp = None
        uv = None
        if (
            FV.WS in self.calc_vars
            or FV.WD in self.calc_vars
            or FV.YAW in self.calc_vars
            or FV.REWS in self.calc_vars
            or FV.REWS2 in self.calc_vars
            or FV.REWS3 in self.calc_vars
        ):
            wd = rpoint_results[FV.WD]
            ws = rpoint_results[FV.WS]
            uvp = wd2uv(wd, ws, axis=-1)
            uv = np.einsum("stpd,p->std", uvp, weights)

        wd = None
        vdone = []
        for v in self.calc_vars:
            if v == FV.WD or v == FV.YAW:
                if wd is None:
                    wd = uv2wd(uv, axis=-1)
                self._set_res(fdata, v, wd, stsel)
                vdone.append(v)

            elif v == FV.WS:
                ws = np.linalg.norm(uv, axis=-1)
                self._set_res(fdata, v, ws, stsel)
                del ws
                vdone.append(v)
        del uv, wd

        if (
            FV.REWS in self.calc_vars
            or FV.REWS2 in self.calc_vars
            or FV.REWS3 in self.calc_vars
        ):
            if stsel is None:
                yaw = fdata[FV.YAW].copy()
            else:
                yaw = fdata[FV.YAW][stsel][:, None]
            nax = wd2uv(yaw, axis=-1)
            wsp = np.einsum("stpd,std->stp", uvp, nax)

            for v in self.calc_vars:
                if v == FV.REWS:
                    rews = np.maximum(np.einsum("stp,p->st", wsp, weights), 0.0)
                    self._set_res(fdata, v, rews, stsel)
                    del rews
                    vdone.append(v)

                elif v == FV.REWS2:
                    # For highly inhomogeneous wind fields
                    # and multiple rotor points some of the uv
                    # vectors may have negative projections onto the
                    # turbine axis direction:
                    if uvp.shape[2] > 1:
                        rews2 = np.sqrt(
                            np.maximum(
                                np.einsum(
                                    "stp,p->st", np.sign(wsp) * wsp**2, weights
                                ),
                                0.0,
                            )
                        )
                    else:
                        rews2 = np.sqrt(np.einsum("stp,p->st", wsp**2, weights))
                    self._set_res(fdata, v, rews2, stsel)
                    del rews2
                    vdone.append(v)

                elif v == FV.REWS3:
                    # For highly inhomogeneous wind fields
                    # and multiple rotor points some of the uv
                    # vectors may have negative projections onto the
                    # turbine axis direction:
                    if uvp.shape[2] > 1:
                        rews3 = np.maximum(
                            np.einsum("stp,p->st", wsp**3, weights), 0.0
                        ) ** (1.0 / 3.0)
                    else:
                        rews3 = (np.einsum("stp,p->st", wsp**3, weights)) ** (
                            1.0 / 3.0
                        )
                    self._set_res(fdata, v, rews3, stsel)
                    del rews3
                    vdone.append(v)

            del wsp
        del uvp

        for v in self.calc_vars:
            if v not in vdone:
                res = np.einsum("stp,p->st", rpoint_results[v], weights)
                self._set_res(fdata, v, res, stsel)
            if copy_to_ambient and v in FV.var2amb:
                fdata[FV.var2amb[v]] = fdata[v].copy()

    def calculate(
        self,
        algo,
        mdata,
        fdata,
        rpoints=None,
        weights=None,
        store_rpoints=False,
        store_rweights=False,
        store_amb_res=False,
        states_turbine=None,
    ):
        """
        Calculate ambient rotor effective results.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        rpoints: numpy.ndarray, optional
            The rotor points, or None for automatic for
            this rotor. Shape: (n_states, n_turbines, n_rpoints, 3)
        weights: numpy.ndarray, optional
            The rotor point weights, or None for automatic
            for this rotor. Shape: (n_rpoints,)
        store_rpoints: bool, optional
            Switch for storing rotor points to mdata
        store_rweights: bool, optional
            Switch for storing rotor point weights to mdata
        store_amb_res: bool, optional
            Switch for storing ambient rotor point reults as they
            come from the states to mdata
        states_turbine: numpy.ndarray of int, optional
            The turbine indices, one per state. Shape: (n_states,)

        Returns
        -------
        results: dict
            results dict. Keys: Variable name str. Values:
            numpy.ndarray with results, shape: (n_states, n_turbines)

        """

        if rpoints is None:
            rpoints = mdata.get(FC.RPOINTS, self.get_rotor_points(algo, mdata, fdata))
        if store_rpoints:
            mdata[FC.RPOINTS] = rpoints
            mdata.dims[FC.RPOINTS] = (FC.STATE, FC.TURBINE, FC.RPOINT, FC.XYH)
            self.data_to_store(FC.RPOINTS, algo, mdata)

        if states_turbine is not None:
            n_states = mdata.n_states
            stsel = (np.arange(n_states), states_turbine)
            rpoints = rpoints[stsel][:, None]
        n_states, n_turbines, n_rpoints, __ = rpoints.shape
        n_points = n_turbines * n_rpoints

        if weights is None:
            weights = mdata.get(FC.RWEIGHTS, self.rotor_point_weights())
        if store_rweights:
            mdata[FC.RWEIGHTS] = weights
            mdata.dims[FC.RWEIGHTS] = (FC.RPOINT,)
            self.data_to_store(FC.RWEIGHTS, algo, mdata)

        svars = algo.states.output_point_vars(algo)
        points = rpoints.reshape(n_states, n_points, 3)
        pdata = {FC.POINTS: points}
        pdims = {FC.POINTS: (FC.STATE, FC.POINT, FC.XYH)}
        pdata.update(
            {v: np.full((n_states, n_points), np.nan, dtype=FC.DTYPE) for v in svars}
        )
        pdims.update({v: (FC.STATE, FC.POINT) for v in svars})
        pdata = Data(pdata, pdims, loop_dims=[FC.STATE, FC.POINT])
        del pdims, points

        sres = algo.states.calculate(algo, mdata, fdata, pdata)
        pdata.update(sres)
        del sres

        rpoint_results = {}
        for v in svars:
            rpoint_results[v] = pdata[v].reshape(n_states, n_turbines, n_rpoints)

        if store_amb_res:
            mdata[FC.AMB_RPOINT_RESULTS] = rpoint_results
            self.data_to_store(FC.AMB_RPOINT_RESULTS, algo, mdata)

        self.eval_rpoint_results(
            algo,
            mdata,
            fdata,
            rpoint_results,
            weights,
            states_turbine,
            copy_to_ambient=True,
        )

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
