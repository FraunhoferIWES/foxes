import numpy as np
from abc import abstractmethod

import foxes.variables as FV
import foxes.constants as FC
from .farm_data_model import FarmDataModel
from foxes.utils import wd2uv, uv2wd, all_subclasses

from .data import TData


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

        self.RPOINTS = self.var("rpoints")
        self.RWEIGHTS = self.var("rweights")
        self.AMBRES = self.var("amb_res")

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
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
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

    def _set_res(self, fdata, v, res, downwind_index):
        """
        Helper function for results setting
        """
        if downwind_index is None:
            fdata[v] = res.copy()
        elif res.shape[1] == 1:
            fdata[v][:, downwind_index] = res[:, 0]
        else:
            raise ValueError(
                f"Rotor model '{self.name}': downwind_index is not None, but results shape for '{v}' has more than one turbine, {res.shape}"
            )

    def eval_rpoint_results(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        weights,
        downwind_index=None,
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
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        weights: numpy.ndarray
            The rotor point weights, shape: (n_rpoints,)
        downwind_index: int, optional
            The index in the downwind order
        copy_to_ambient: bool
            If `True`, the fdata results are copied to ambient
            variables after calculation

        """
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
            wd = tdata[FV.WD]
            ws = tdata[FV.WS]
            uvp = wd2uv(wd, ws, axis=-1)
            uv = np.einsum("stpd,p->std", uvp, weights)

        wd = None
        vdone = []
        for v in self.calc_vars:
            if v == FV.WD or v == FV.YAW:
                if wd is None:
                    wd = uv2wd(uv, axis=-1)
                self._set_res(fdata, v, wd, downwind_index)
                vdone.append(v)

            elif v == FV.WS:
                ws = np.linalg.norm(uv, axis=-1)
                self._set_res(fdata, v, ws, downwind_index)
                del ws
                vdone.append(v)
        del uv, wd

        if (
            FV.REWS in self.calc_vars
            or FV.REWS2 in self.calc_vars
            or FV.REWS3 in self.calc_vars
        ):
            if downwind_index is None:
                yaw = fdata[FV.YAW].copy()
            else:
                yaw = fdata[FV.YAW][:, downwind_index, None]
            nax = wd2uv(yaw, axis=-1)
            wsp = np.einsum("stpd,std->stp", uvp, nax)

            for v in self.calc_vars:
                if v == FV.REWS:
                    rews = np.maximum(np.einsum("stp,p->st", wsp, weights), 0.0)
                    self._set_res(fdata, v, rews, downwind_index)
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
                                np.einsum("stp,p->st", np.sign(wsp) * wsp**2, weights),
                                0.0,
                            )
                        )
                    else:
                        rews2 = np.sqrt(np.einsum("stp,p->st", wsp**2, weights))
                    self._set_res(fdata, v, rews2, downwind_index)
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
                        rews3 = (np.einsum("stp,p->st", wsp**3, weights)) ** (1.0 / 3.0)
                    self._set_res(fdata, v, rews3, downwind_index)
                    del rews3
                    vdone.append(v)

            del wsp
        del uvp

        for v in self.calc_vars:
            if v not in vdone:
                res = np.einsum("stp,p->st", tdata[v], weights)
                self._set_res(fdata, v, res, downwind_index)
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
        downwind_index=None,
    ):
        """
        Calculate ambient rotor effective results.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
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
        downwind_index: int, optional
            Only compute for index in the downwind order

        Returns
        -------
        results: dict
            results dict. Keys: Variable name str. Values:
            numpy.ndarray with results, shape: (n_states, n_turbines)

        """

        if rpoints is None:
            rpoints = mdata.get(self.RPOINTS, self.get_rotor_points(algo, mdata, fdata))
        if store_rpoints:
            mdata[self.RPOINTS] = rpoints
            mdata.dims[self.RPOINTS] = (FC.STATE, FC.TURBINE, FC.TPOINT, FC.XYH)
            self.data_to_store(self.RPOINTS, algo, mdata)

        if downwind_index is not None:
            rpoints = rpoints[:, downwind_index, None]

        if weights is None:
            weights = mdata.get(FC.TWEIGHTS, self.rotor_point_weights())
        if store_rweights:
            mdata[self.RWEIGHTS] = weights
            mdata.dims[self.RWEIGHTS] = (FC.TPOINT,)
            self.data_to_store(self.RWEIGHTS, algo, mdata)

        tdata = TData.from_tpoints(rpoints, weights)
        svars = algo.states.output_point_vars(algo)
        for v in svars:
            tdata.add(
                v,
                data=np.full_like(rpoints[..., 0], np.nan),
                dims=(FC.STATE, FC.TARGET, FC.TPOINT),
            )

        sres = algo.states.calculate(algo, mdata, fdata, tdata)
        tdata.update(sres)

        if store_amb_res:
            mdata[self.AMBRES] = sres.copy()
            self.data_to_store(self.AMBRES, algo, mdata)

        self.eval_rpoint_results(
            algo,
            mdata,
            fdata,
            tdata,
            weights,
            downwind_index,
            copy_to_ambient=True,
        )

        return {v: fdata[v] for v in self.output_farm_vars(algo)}

    @classmethod
    def new(cls, rmodel_type, *args, **kwargs):
        """
        Run-time rotor model factory.

        Parameters
        ----------
        rmodel_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """

        if rmodel_type is None:
            return None

        allc = all_subclasses(cls)
        found = rmodel_type in [scls.__name__ for scls in allc]

        if found:
            for scls in allc:
                if scls.__name__ == rmodel_type:
                    return scls(*args, **kwargs)

        else:
            estr = "Rotor model type '{}' is not defined, available types are \n {}".format(
                rmodel_type, sorted([i.__name__ for i in allc])
            )
            raise KeyError(estr)
