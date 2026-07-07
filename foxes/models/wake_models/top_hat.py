from __future__ import annotations

import numpy as np
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from foxes.models.wake_models.axisymmetric import AxisymmetricWakeModel
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData, Model
    from foxes.core.axial_induction_model import AxialInductionModel


class TopHatWakeModel(AxisymmetricWakeModel):
    """
    Abstract base class for top-hat wake models.

    Parameters
    ----------
    induction: foxes.core.AxialInductionModel or str
        The induction model

    :group: models.wake_models

    """

    def __init__(
        self, *args: Any, induction: AxialInductionModel | str = "Betz", **kwargs: Any
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Additional parameters for the base class
        induction: foxes.core.AxialInductionModel or str
            The induction model
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(*args, **kwargs)
        self.induction = induction

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        smdls = super().sub_models()
        if not isinstance(self.induction, str):
            smdls.append(self.induction)
        return smdls

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        loaded_data: dict, optional
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force: bool
            Overwrite existing data
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data: dict
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        if isinstance(self.induction, str):
            self.induction = algo.mbook.axial_induction[self.induction]
        return super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

    def new_wake_deltas(
        self, algo: Algorithm, mdata: MData, fdata: FData, tdata: TData
    ) -> dict[str, np.ndarray]:
        """
        Creates new empty wake delta arrays.

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

        Returns
        -------
        wake_deltas: dict
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_targets, n_tpoints, ...)

        """
        if self.has_uv:
            duv = np.zeros(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints, 2),
                dtype=config.dtype_double,
            )
            return {FV.UV: duv}
        else:
            dws = np.zeros(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints),
                dtype=config.dtype_double,
            )
            return {FV.WS: dws}

    @abstractmethod
    def calc_wake_radius(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        x: np.ndarray,
        ct: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the wake radius, depending on x only (not r).

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
            The index in the downwind order
        x: numpy.ndarray
            The x values, shape: (n_states, n_targets)
        ct: numpy.ndarray
            The ct values of the wake-causing turbines,
            shape: (n_states, n_targets)

        Returns
        -------
        wake_r: numpy.ndarray
            The wake radii, shape: (n_states, n_targets)

        """
        pass

    @abstractmethod
    def calc_centreline(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        st_sel: np.ndarray,
        x: np.ndarray,
        wake_r: np.ndarray,
        ct: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Calculate centre line results of wake deltas.

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
            The index in the downwind order
        st_sel: numpy.ndarray of bool
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)
        x: numpy.ndarray
            The x values, shape: (n_st_sel,)
        wake_r: numpy.ndarray
            The wake radii, shape: (n_st_sel,)
        ct: numpy.ndarray
            The ct values of the wake-causing turbines,
            shape: (n_st_sel,)

        Returns
        -------
        cl_del: dict
            The centre line wake deltas. Key: variable name str,
            varlue: numpy.ndarray, shape: (n_st_sel,)

        """
        pass

    def calc_wakes_x_r(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        x: np.ndarray,
        r: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Calculate wake deltas.

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
            The index in the downwind order
        x: numpy.ndarray
            The x values, shape: (n_states, n_targets)
        r: numpy.ndarray
            The radial values for each x value, shape:
            (n_states, n_targets, n_yz_per_target)

        Returns
        -------
        wdeltas: dict
            The wake deltas. Key: variable name str,
            value: numpy.ndarray, shape: (n_st_sel, n_r_per_x)
        st_sel: numpy.ndarray of bool
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        """
        ct = self.get_data(
            FV.CT,
            FC.STATE_TARGET,
            lookup="w",
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            algo=algo,
            upcast=True,
        )

        wake_r = self.calc_wake_radius(algo, mdata, fdata, tdata, downwind_index, x, ct)

        wdeltas = {}
        st_sel = (x > 1e-8) & (ct > 1e-8) & np.any(r < wake_r[:, :, None], axis=2)
        if np.any(st_sel):
            x = x[st_sel]
            r = r[st_sel]
            ct = ct[st_sel]
            wake_r = wake_r[st_sel]

            cl_del = self.calc_centreline(
                algo, mdata, fdata, tdata, downwind_index, st_sel, x, wake_r, ct
            )

            isin = r < wake_r[:, None]
            for v, wdel in cl_del.items():
                wdeltas[v] = np.where(isin, wdel[:, None], 0.0)

        if self.affects_ws and FV.WS in wdeltas:
            # wake deflection causes wind vector rotation:
            if FC.WDEFL_ROT_ANGLE in tdata:
                dwd_defl = tdata[FC.WDEFL_ROT_ANGLE]
                if FV.WD not in wdeltas:
                    wdeltas[FV.WD] = np.zeros_like(wdeltas[FV.WS])
                    wdeltas[FV.WD][:] = dwd_defl[st_sel]
                else:
                    wdeltas[FV.WD] += dwd_defl[st_sel]

            # wake deflection causes wind speed reduction:
            if FC.WDEFL_DWS_FACTOR in tdata:
                dws_defl = tdata[FC.WDEFL_DWS_FACTOR]
                wdeltas[FV.WS] *= dws_defl[st_sel]

        return wdeltas, st_sel
