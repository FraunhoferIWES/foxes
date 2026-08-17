from __future__ import annotations

from abc import abstractmethod
import numpy as np
from typing import TYPE_CHECKING, Any, cast

from foxes.utils import new_instance, wd2uv, uv2wd
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

from .model import Model
from .data import TData

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData
    from foxes.core.wake_model import WakeModel


class PartialWakesModel(Model):
    """
    Abstract base class for partial wakes models.

    Partial wakes models compute wake effects
    for rotor effective quantities.

    Attributes
    ----------
    wake_models
        The wake model selection
    wake_frame
        The wake frame

    :group: core

    """

    def check_wmodel(self, wmodel: WakeModel, error: bool = True) -> bool:
        """
        Checks the wake model type

        Parameters
        ----------
        wmodel
            The wake model to be tested
        error
            Flag for raising TypeError

        Returns
        -------
        chk
            True if wake model is compatible

        """
        return True

    @abstractmethod
    def get_wake_points(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the wake calculation points, and their
        weights.

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
        rpoints
            The wake calculation points, shape:
            (n_states, n_turbines, n_tpoints, 3)
        rweights
            The target point weights, shape: (n_tpoints,)

        """
        pass

    def get_initial_tdata(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        amb_rotor_res: dict[str, np.ndarray],
        rotor_weights: np.ndarray,
        wmodels: list[WakeModel],
    ) -> TData:
        """
        Creates the initial target data object

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        amb_rotor_res
            The ambient results at rotor points,
            keyed by variable name, with array values
            of shape: (n_states, n_turbines, n_rotor_points)
        rotor_weights
            The rotor point weights, shape: (n_rotor_points,)
        wmodels
            The wake models for this partial wake model

        Returns
        -------
        tdata
            The target point data for the wake points

        """
        tpoints, tweights = self.get_wake_points(algo, mdata, fdata)
        tdata = cast(TData, TData.from_tpoints(tpoints, tweights))

        self.update_tdata(
            algo,
            mdata,
            fdata,
            tdata,
            amb_rotor_res,
            rotor_weights,
            wmodels,
            downwind_index=None,
        )

        return tdata

    def update_tdata(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        amb_rotor_res: dict[str, np.ndarray],
        rotor_weights: np.ndarray,
        wmodels: list[WakeModel],
        downwind_index: int | None = None,
    ) -> None:
        """
        Updates tdata on the fly during wake calculations.

        This method can be used to update the target data on the fly
        during the wake calculations, after new rotor model calculations
        have been performed.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data for the wake points
        amb_rotor_res
            The ambient results at rotor points,
            keyed by variable name, with array values
            of shape: (n_states, n_turbines, n_rotor_points)
        rotor_weights
            The rotor point weights, shape: (n_rotor_points,)
        wmodels
            The wake models for this partial wake model
        downwind_index
            The downwind index of the updated turbine

        """
        # prepare:
        s = np.s_[:] if downwind_index is None else np.s_[:, downwind_index, ...]

        # map wind data:
        if FV.WD in amb_rotor_res or FV.WS in amb_rotor_res:
            assert FV.WD in amb_rotor_res and FV.WS in amb_rotor_res, (
                "Require both wind direction and speed in ambient rotor results."
            )
            uv = wd2uv(amb_rotor_res[FV.WD][s], amb_rotor_res[FV.WS][s])
            uv = np.stack(
                [
                    self.map_rotor_results(
                        algo,
                        mdata,
                        fdata,
                        tdata,
                        FV.U,
                        uv[..., 0],
                        rotor_weights,
                        downwind_index,
                    ),
                    self.map_rotor_results(
                        algo,
                        mdata,
                        fdata,
                        tdata,
                        FV.V,
                        uv[..., 1],
                        rotor_weights,
                        downwind_index,
                    ),
                ],
                axis=-1,
            )

            if downwind_index is None:
                tdata.add(
                    FV.AMB_WD,
                    uv2wd(uv),
                    dims=(FC.STATE, FC.TARGET, FC.TPOINT),
                )
                tdata.add(
                    FV.AMB_WS,
                    np.linalg.norm(uv, axis=-1),
                    dims=(FC.STATE, FC.TARGET, FC.TPOINT),
                )
            else:
                tdata[FV.AMB_WD][s] = uv2wd(uv)
                tdata[FV.AMB_WS][s] = np.linalg.norm(uv, axis=-1)

            for wmodel in wmodels:
                if wmodel.has_uv:
                    if downwind_index is None:
                        tdata.add(
                            FV.AMB_UV,
                            uv,
                            dims=(FC.STATE, FC.TARGET, FC.TPOINT, FC.XY),
                        )
                    else:
                        tdata[FV.AMB_UV][s] = uv
                    break

        # map rotor point results onto target points:
        for v, d in amb_rotor_res.items():
            if v not in [FV.WS, FV.WD, FV.U, FV.V, FV.UV]:
                w = FV.var2amb.get(v, v)
                if downwind_index is None:
                    tdata.add(
                        w,
                        self.map_rotor_results(
                            algo,
                            mdata,
                            fdata,
                            tdata,
                            v,
                            d[s],
                            rotor_weights,
                            downwind_index=downwind_index,
                        ),
                        dims=(FC.STATE, FC.TARGET, FC.TPOINT),
                    )
                else:
                    tdata[w][s] = self.map_rotor_results(
                        algo,
                        mdata,
                        fdata,
                        tdata,
                        v,
                        d[s],
                        rotor_weights,
                        downwind_index=downwind_index,
                    )

    def map_rotor_results(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        variable: str,
        rotor_res: np.ndarray,
        rotor_weights: np.ndarray,
        downwind_index: int | None = None,
    ) -> np.ndarray:
        """
        Map ambient rotor point results onto target points.

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
        variable
            The variable name to map
        rotor_res
            The results at rotor points, shape:
            (n_states, n_turbines, n_rotor_points) if downwind_index is None,
            otherwise shape: (n_states, n_rotor_points)
        rotor_weights
            The rotor point weights, shape: (n_rotor_points,)
        downwind_index
            The downwind index of the updated turbine,
            if None, maps for all turbines

        Returns
        -------
        res
            The mapped results at target points, shape:
            (n_states, n_targets, n_tpoints) if downwind_index is None,
            otherwise shape: (n_states, n_tpoints)

        """
        if (
            downwind_index is None
            and len(rotor_res.shape) == 3
            and rotor_res.shape[:2]
            == (
                tdata.n_states,
                tdata.n_targets,
            )
        ):
            q: np.ndarray = np.zeros(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints),
                dtype=config.dtype_double,
            )
            if rotor_res.shape[2] == 1:
                q[:] = rotor_res
            else:
                q[:] = np.einsum("str,r->st", rotor_res, rotor_weights)[:, :, None]
            return q

        elif (
            downwind_index is not None
            and len(rotor_res.shape) == 2
            and rotor_res.shape[0] == tdata.n_states
        ):
            q = np.zeros(
                (tdata.n_states, tdata.n_tpoints),
                dtype=config.dtype_double,
            )
            if rotor_res.shape[1] == 1:
                q[:] = rotor_res[:, 0][:, None]
            else:
                q[:] = np.einsum("sr,r->s", rotor_res, rotor_weights)[:, None]

            return q

        else:
            raise ValueError(
                f"Partial wakes '{self.name}': Incompatible shape '{rotor_res.shape}' for variable '{variable}' in rotor results."
            )

    def new_wake_deltas(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        wmodel: WakeModel,
    ) -> dict[str, np.ndarray]:
        """
        Creates new initial wake deltas, filled
        with zeros.

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
        wmodel
            The wake model

        Returns
        -------
        wake_deltas
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_turbines, n_tpoints, ...)

        """
        return wmodel.new_wake_deltas(algo, mdata, fdata, tdata)

    def contribute(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        wake_deltas: dict[str, np.ndarray],
        wmodel: WakeModel,
    ) -> None:
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

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
        wake_deltas
            The wake deltas. Key: variable name,
            values are arrays with shape
            (n_states, n_targets, n_tpoints, ...)
        wmodel
            The wake model

        """
        wake_frame = algo.wake_frame
        wcoos = wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)
        wmodel.contribute(algo, mdata, fdata, tdata, downwind_index, wcoos, wake_deltas)

    @abstractmethod
    def finalize_wakes(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        rpoint_weights: np.ndarray,
        wake_deltas: dict[str, np.ndarray],
        wmodel: WakeModel,
        downwind_index: int,
    ) -> dict[str, np.ndarray]:
        """
        Updates the wake_deltas at the selected target
        downwind index.

        Modifies wake_deltas on the fly.

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
        rpoint_weights
            The rotor point weights, shape: (n_rotor_points,)
        wake_deltas
            The wake deltas. Key: variable name,
            value: np.ndarray of shape
            (n_states, n_turbines, n_tpoints)
        wmodel
            The wake model
        downwind_index
            The index in the downwind order

        Returns
        -------
        final_wake_deltas
            The final wake deltas at the selected downwind
            turbines. Key: variable name, value: np.ndarray
            of shape (n_states, n_rotor_points)

        """
        pass

    @classmethod
    def new(
        cls,
        pwakes_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> PartialWakesModel:
        """
        Run-time partial wakes model factory.

        Parameters
        ----------
        pwakes_type
            The selected derived class name
        args
            Additional parameters for the constructor
        kwargs
            Additional parameters for the constructor

        """
        return new_instance(cls, pwakes_type, *args, **kwargs)
