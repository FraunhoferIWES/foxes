from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, cast

from foxes.config import config
from foxes.utils import wd2uv, uv2wd, new_instance

import foxes.variables as FV
import foxes.constants as FC

from .data import TData
from .farm_data_model import FarmDataModel

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData


class RotorModel(FarmDataModel):
    """
    Abstract base class of rotor models.

    Rotor models calculate ambient farm data from
    states, and provide rotor points and weights
    for the calculation of rotor effective quantities.
    """

    def __init__(self, calc_vars: list[str] | None = None) -> None:
        """
        Parameters
        ----------
        calc_vars
            The variables calculated by the model. Their ambients are added
            automatically.
        """
        super().__init__()
        self.calc_vars = calc_vars

    @abstractmethod
    def input_variables(self) -> list[str]:
        """
        Return the input variables required by the model.

        Returns
        -------
        input_vars
            The input variable names.

        """
        pass

    def output_farm_vars(self, algo: Algorithm) -> list[str]:
        """
        Return the variables modified by the model.

        Parameters
        ----------
        algo
            The calculation algorithm.

        Returns
        -------
        output_vars
            The output variable names.

        """
        states = algo.states
        farm_controller = algo.farm_controller

        if self.calc_vars is None:
            calc_vars: list[str]
            vrs = states.output_point_vars(algo)
            assert FV.WEIGHT not in vrs, (
                f"Rotor '{self.name}': States '{states.name}' output_point_vars contain '{FV.WEIGHT}', please remove"
            )

            if FV.WS in vrs:
                calc_vars = [FV.REWS] + [v for v in vrs if v != FV.WS]
            else:
                calc_vars = list(vrs)

            if farm_controller.needs_rews2() and FV.REWS2 not in calc_vars:
                calc_vars.append(FV.REWS2)
            if farm_controller.needs_rews3() and FV.REWS3 not in calc_vars:
                calc_vars.append(FV.REWS3)

            calc_vars = sorted(calc_vars)
        else:
            calc_vars = list(self.calc_vars)

        calc_vars = [v for v in calc_vars if v not in self.input_variables()]

        if FV.WEIGHT not in calc_vars:
            calc_vars.append(FV.WEIGHT)

        self.calc_vars = calc_vars
        return calc_vars

    @abstractmethod
    def n_rotor_points(self) -> int:
        """
        Return the number of rotor points.

        Returns
        -------
        n_rpoints
            The number of rotor points.

        """
        pass

    @abstractmethod
    def rotor_point_weights(self) -> np.ndarray:
        """
        Return the weights of the rotor points.

        Returns
        -------
        weights
            The rotor-point weights, which sum to one and have shape
            ``(n_rpoints,)``.

        """
        pass

    @abstractmethod
    def design_points(self) -> np.ndarray:
        """
        Return the rotor-model design points.

        Design points are formulated in rotor-plane ``(x, y, z)`` coordinates in
        the rotor frame, such that ``(0, 0, 0)`` is the center point,
        ``(1, 0, 0)`` is the point radius times ``n_rotor_axis``,
        ``(0, 1, 0)`` is the point radius times ``n_rotor_side``, and
        ``(0, 0, 1)`` is the point radius times ``n_rotor_up``.

        Returns
        -------
        dpoints
            The design points with shape ``(n_points, 3)``.

        """
        pass

    def get_rotor_points(
        self, algo: Algorithm, mdata: MData, fdata: FData
    ) -> np.ndarray:
        """
        Calculate rotor points from design points.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            The model data.
        fdata
            The farm data.

        Returns
        -------
        points
            The rotor points with shape ``(n_states, n_turbines, n_rpoints, 3)``.

        """

        n_states = mdata.n_states
        n_points = self.n_rotor_points()
        n_turbines = algo.n_turbines
        dpoints = self.design_points()
        D = fdata[FV.D]

        rax: np.ndarray = np.zeros(
            (n_states, n_turbines, 3, 3), dtype=config.dtype_double
        )
        n = rax[:, :, 0, 0:2]
        m = rax[:, :, 1, 0:2]
        n[:] = wd2uv(fdata[FV.YAW], axis=-1)
        m[:] = np.stack([-n[:, :, 1], n[:, :, 0]], axis=-1)
        rax[:, :, 2, 2] = 1

        points: np.ndarray = np.zeros(
            (n_states, n_turbines, n_points, 3), dtype=config.dtype_double
        )
        points[:] = fdata[FV.TXYH][:, :, None, :]
        points[:] += (
            0.5 * D[:, :, None, None] * np.einsum("stad,pa->stpd", rax, dpoints)
        )

        return points

    def _set_res(
        self,
        fdata: FData,
        v: str,
        res: np.ndarray,
        downwind_index: int | None,
    ) -> None:
        """
        Helper function for results setting
        """
        if downwind_index is None:
            fdata[v] = res.copy()
        elif res.shape[1] == 1:
            fdata[v][:, downwind_index] = res[:, 0]
        else:
            fdata[v, downwind_index] = res[:, downwind_index]

    def eval_rpoint_results(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        rpoint_weights: np.ndarray,
        downwind_index: int | None = None,
        copy_to_ambient: bool = False,
        set_wd: bool = False,
    ) -> None:
        """
        Evaluate rotor-point results.

        This function modifies ``fdata`` for either all turbines or one turbine
        per state, depending on the ``states_turbine`` setting. In the latter
        case, the turbine dimension of the rotor-point results is expected to have
        size one.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            The model data.
        fdata
            The farm data.
        tdata
            The target-point data.
        rpoint_weights
            The rotor-point weights with shape ``(n_rpoints,)``.
        downwind_index
            The index in the downwind order.
        copy_to_ambient
            If ``True``, the ``fdata`` results are copied to ambient variables
            after calculation.
        set_wd
            If ``True``, the wind direction is updated.

        """
        if self.calc_vars is None:
            self.output_farm_vars(algo)
        assert self.calc_vars is not None

        for v in [FV.REWS2, FV.REWS3]:
            if v in fdata and v not in self.calc_vars:
                self.calc_vars.append(v)

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
            uv = np.einsum("stpd,p->std", uvp, rpoint_weights)

        wd = None
        vdone = []
        for v in self.calc_vars:
            if (set_wd and v == FV.WD) or v == FV.YAW:
                if wd is None:
                    assert uv is not None
                    wd = uv2wd(uv, axis=-1)
                self._set_res(fdata, v, wd, downwind_index)
                vdone.append(v)
            elif v == FV.WS:
                assert uv is not None
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
            assert uvp is not None
            if downwind_index is None:
                yaw = fdata[FV.YAW].copy()
            else:
                yaw = fdata[FV.YAW][:, downwind_index, None]
            nax = wd2uv(yaw, axis=-1)
            wsp = np.einsum("stpd,std->stp", uvp, nax)

            for v in self.calc_vars:
                if v == FV.REWS:
                    rews = np.maximum(np.einsum("stp,p->st", wsp, rpoint_weights), 0.0)
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
                                np.einsum(
                                    "stp,p->st", np.sign(wsp) * wsp**2, rpoint_weights
                                ),
                                0.0,
                            )
                        )
                    else:
                        rews2 = np.sqrt(np.einsum("stp,p->st", wsp**2, rpoint_weights))
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
                            np.einsum("stp,p->st", wsp**3, rpoint_weights), 0.0
                        ) ** (1.0 / 3.0)
                    else:
                        rews3 = (np.einsum("stp,p->st", wsp**3, rpoint_weights)) ** (
                            1.0 / 3.0
                        )
                    self._set_res(fdata, v, rews3, downwind_index)
                    del rews3
                    vdone.append(v)

            del wsp
        del uvp

        for v in self.calc_vars:
            if not (v == FV.WD and not set_wd):
                if (
                    v not in vdone
                    and (
                        fdata[v].shape[1] > 1
                        or downwind_index is None
                        or downwind_index == 0
                    )
                    and not (v == FV.WD and not set_wd)
                ):
                    res = np.einsum("stp,p->st", tdata[v], rpoint_weights)
                    self._set_res(fdata, v, res, downwind_index)
                if copy_to_ambient and v in FV.var2amb:
                    fdata[FV.var2amb[v]] = fdata[v].copy()

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        rpoints: np.ndarray | None = None,
        rpoint_weights: np.ndarray | None = None,
        store: bool = False,
        downwind_index: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Calculate ambient rotor-effective results.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            The model data.
        fdata
            The farm data.
        rpoints
            The rotor points, or ``None`` for automatic selection for this rotor.
            Shape is ``(n_states, n_turbines, n_rpoints, 3)``.
        rpoint_weights
            The rotor-point weights, or ``None`` for automatic selection for this
            rotor. Shape is ``(n_rpoints,)``.
        store
            Flag for storing ambient rotor-point results.
        downwind_index
            Only compute for the selected index in the downwind order.

        Returns
        -------
        results
            A dictionary of results keyed by variable name. Values are NumPy
            arrays with shape ``(n_states, n_turbines)``.

        """
        self.ensure_output_vars(algo, fdata)

        if rpoints is None:
            rpoints = mdata.get(
                FC.ROTOR_POINTS, self.get_rotor_points(algo, mdata, fdata)
            )
        assert isinstance(rpoints, np.ndarray)
        if downwind_index is not None:
            rpoints = rpoints[:, downwind_index, None]
        if rpoint_weights is None:
            rpoint_weights = mdata.get_item(FC.TWEIGHTS, self.rotor_point_weights())
            algo.add_to_chunk_store(
                FC.ROTOR_WEIGHTS,
                rpoint_weights,
                dims=(FC.ROTOR_POINT,),
                mdata=mdata,
            )
        assert isinstance(rpoint_weights, np.ndarray)

        tdata = cast(TData, TData.from_tpoints(rpoints, rpoint_weights))
        states = algo.states
        svars = states.output_point_vars(algo)
        for v in svars:
            tdata.add(
                v,
                data=np.full_like(rpoints[..., 0], np.nan),
                dims=(FC.STATE, FC.TARGET, FC.TPOINT),
            )

        sres = states.calculate(algo, mdata, fdata, tdata)
        tdata.update(sres)
        if FV.WEIGHT not in tdata:
            raise KeyError(
                f"Rotor '{self.name}': States '{states.name}' failed to provide '{FV.WEIGHT}' in tdata"
            )

        if store:
            s = None if downwind_index is None else np.s_[:, downwind_index, ...]
            algo.add_to_chunk_store(
                FC.ROTOR_POINTS,
                rpoints if downwind_index is None else rpoints[:, 0, ...],
                dims=(FC.STATE, FC.TURBINE, FC.ROTOR_POINT, FC.XYH),
                mdata=mdata,
                subset=s,
            )
            algo.add_to_chunk_store(
                FC.AMB_ROTOR_RES,
                sres,
                dims=(FC.STATE, FC.TURBINE, FC.ROTOR_POINT),
                mdata=mdata,
                subset=s,
            )
            if downwind_index is None or (
                (weight_res := algo.get_from_chunk_store(FC.WEIGHT_RES, mdata=mdata))
                is not None
                and weight_res.shape[1] > 1
            ):
                algo.add_to_chunk_store(
                    FC.WEIGHT_RES,
                    tdata[FV.WEIGHT]
                    if downwind_index is None
                    else tdata[FV.WEIGHT][:, 0, ...],
                    dims=(FC.STATE, FC.TURBINE, FC.ROTOR_POINT),
                    mdata=mdata,
                    subset=s,
                )

        self.eval_rpoint_results(
            algo,
            mdata,
            fdata,
            tdata,
            rpoint_weights,
            downwind_index,
            copy_to_ambient=True,
        )

        return {v: fdata[v] for v in self.output_farm_vars(algo)}

    @classmethod
    def new(cls, rmodel_type: str, *args: Any, **kwargs: Any) -> RotorModel:
        """
        Run-time rotor model factory.

        Parameters
        ----------
        rmodel_type
            The selected derived class name
        args
            Additional parameters for constructor
        kwargs
            Additional parameters for constructor

        """
        return cast(RotorModel, new_instance(cls, rmodel_type, *args, **kwargs))
