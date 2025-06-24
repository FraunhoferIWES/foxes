import numpy as np
from abc import abstractmethod

from foxes.models.wake_models.axisymmetric import AxisymmetricWakeModel
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC


class TopHatWakeModel(AxisymmetricWakeModel):
    """
    Abstract base class for top-hat wake models.

    Parameters
    ----------
    induction: foxes.core.AxialInductionModel or str
        The induction model

    :group: models.wake_models

    """

    def __init__(self, *args, induction="Betz", **kwargs):
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

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return super().sub_models() + [self.induction]

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        if isinstance(self.induction, str):
            self.induction = algo.mbook.axial_induction[self.induction]
        super().initialize(algo, verbosity, force)

    def new_wake_deltas(self, algo, mdata, fdata, tdata):
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
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        x,
        ct,
    ):
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
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        st_sel,
        x,
        wake_r,
        ct,
    ):
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
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        x,
        r,
    ):
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

        if self.affects_ws:

            # wake deflection causes wind vector rotation:
            if FC.WDEFL_ROT_ANGLE in tdata:
                assert FV.WS in wdeltas, f"Wake model '{self.name}': Expecting '{FV.WS}' in wdeltas, found {list(wdeltas.keys())}"
                dwd_defl = tdata[FC.WDEFL_ROT_ANGLE]
                if FV.WD not in wdeltas:
                    wdeltas[FV.WD] = np.zeros_like(wdeltas[FV.WS])
                    wdeltas[FV.WD][:] = dwd_defl[st_sel]
                else:
                    wdeltas[FV.WD] += dwd_defl[st_sel]
            
            # wake deflection causes wind speed reduction:
            if FC.WDEFL_DWS_FACTOR in tdata:
                assert FV.WS in wdeltas, f"Wake model '{self.name}': Expecting '{FV.WS}' in wdeltas, found {list(wdeltas.keys())}"
                dws_defl = tdata[FC.WDEFL_DWS_FACTOR]
                wdeltas[FV.WS] *= dws_defl[st_sel]

        return wdeltas, st_sel
