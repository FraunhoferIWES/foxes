import numpy as np

from foxes.core import WindVectorWakeSuperposition
from foxes.utils import wd2uv, uv2wd, delta_wd
import foxes.variables as FV
import foxes.constants as FC


class WindVectorLinear(WindVectorWakeSuperposition):
    """
    Linear superposition of wind deficit vector results

    Attributes
    ----------
    scale_amb: bool
        Flag for scaling wind deficit with ambient wind speed
        instead of waked wind speed

    :group: models.wake_superpositions

    """

    def __init__(self, scale_amb=False):
        """
        Constructor.

        Parameters
        ----------
        scale_amb: bool
            Flag for scaling wind deficit with ambient wind speed
            instead of waked wind speed

        """
        super().__init__()

        self.scale_amb = scale_amb

    def __repr__(self):
        a = f"scale_amb={self.scale_amb}, lim_low={self.lim_low}, lim_high={self.lim_high}"
        return f"{type(self).__name__}({a})"

    def input_farm_vars(self, algo):
        """
        The variables which are needed for running
        the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        input_vars: list of str
            The input variable names

        """
        return [FV.AMB_REWS] if self.scale_amb else [FV.REWS]

    def wdeltas_ws2uv(self, algo, fdata, tdata, downwind_index, wdeltas, st_sel):
        """
        Transform results from wind speed to wind vector data
        
        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        downwind_index: int
            The index of the wake causing turbine
            in the downwind order
        wdeltas: dict
            The wake deltas. Key: variable name str,
            value: numpy.ndarray, shape: (n_st_sel, n_tpoints)
        st_sel: numpy.ndarray of bool
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)
        
        Returns
        -------
        wdeltas: dict
            The wake deltas. Key: variable name str,
            value: numpy.ndarray, now respecting has_uv flag

        """
        if FV.AMB_UV not in tdata:
            tdata[FV.AMB_UV] = wd2uv(tdata[FV.AMB_WD], tdata[FV.AMB_WS])
        if FV.UV not in wdeltas:
            assert FV.WS in wdeltas, f"{self.name}: Expecting '{FV.WS}' in wdeltas, got {list(wdeltas.keys())}"
            scale = self.get_data(
                FV.AMB_REWS if self.scale_amb else FV.REWS,
                FC.STATE_TARGET_TPOINT,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=False,
                selection=st_sel,
            )
            ws0 = tdata[FV.AMB_WS][st_sel]
            wd0 = tdata[FV.AMB_WD][st_sel]
            dws = scale * wdeltas.pop(FV.WS)
            dwd = wdeltas.pop(FV.WD, 0)
            wdeltas[FV.UV] = wd2uv(wd0 + dwd, ws0 + dws) - tdata[FV.AMB_UV][st_sel]

        return wdeltas
    
    def wdeltas_uv2ws(self, algo, fdata, tdata, downwind_index, wdeltas, st_sel):
        """
        Transform results from wind vector to wind speed data
        
        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        downwind_index: int
            The index of the wake causing turbine
            in the downwind order
        wdeltas: dict
            The wake deltas. Key: variable name str,
            value: numpy.ndarray, shape: (n_st_sel, n_tpoints)
        st_sel: numpy.ndarray of bool
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)
        
        Returns
        -------
        wdeltas: dict
            The wake deltas. Key: variable name str,
            value: numpy.ndarray, now respecting has_uv flag

        """     
        if FV.UV in wdeltas:
            scale = self.get_data(
                FV.AMB_REWS if self.scale_amb else FV.REWS,
                FC.STATE_TARGET_TPOINT,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=False,
                selection=st_sel,
            )
            ws0 = tdata[FV.AMB_WS][st_sel]
            wd0 = tdata[FV.AMB_WD][st_sel]
            uv = tdata[FV.AMB_UV][st_sel] + wdeltas.pop(FV.UV)
            wdeltas[FV.WD] = delta_wd(wd0, uv2wd(uv))
            wdeltas[FV.WS] = (np.linalg.norm(uv, axis=-1) - ws0) / scale
        
        return wdeltas
    
    def add_wake_vector(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        st_sel,
        wake_delta_uv,
        wake_model_result_uv,
    ):
        """
        Add a wake delta vector to previous wake deltas,
        at rotor points.

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
        st_sel: numpy.ndarray of bool
            The selection of targets, shape: (n_states, n_targets)
        wake_delta_uv: numpy.ndarray
            The original wind vector wake deltas, shape:
            (n_states, n_targets, n_tpoints, 2)
        wake_model_result_uv: numpy.ndarray
            The new wind vector wake deltas of the selected rotors,
            shape: (n_st_sel, n_tpoints, 2, ...)

        Returns
        -------
        wdelta_uv: numpy.ndarray
            The updated wind vector wake deltas, shape:
            (n_states, n_targets, n_tpoints, ...)

        """

        if np.any(st_sel):
            wake_delta_uv[st_sel] += wake_model_result_uv
        
        return wake_delta_uv

    def calc_final_wake_delta_uv(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        wake_delta_uv,
    ):
        """
        Calculate the final wind vector wake delta after adding all
        contributions.

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
        wake_delta_uv: numpy.ndarray
            The original wind vector wake deltas, shape:
            (n_states, n_targets, n_tpoints, 2)

        Returns
        -------
        final_wake_delta_ws: numpy.ndarray
            The final wind speed wake delta, which will be added to 
            the ambient results by simple plus operation. Shape:
            (n_states, n_targets, n_tpoints)
        final_wake_delta_wd: numpy.ndarray
            The final wind direction wake delta, which will be added to 
            the ambient results by simple plus operation. Shape:
            (n_states, n_targets, n_tpoints)

        """
        if FV.AMB_UV not in tdata:
            tdata[FV.AMB_UV] = wd2uv(tdata[FV.AMB_WD], tdata[FV.AMB_WS])

        uv = tdata[FV.AMB_UV] + wake_delta_uv
        dwd = delta_wd(tdata[FV.AMB_WD], uv2wd(uv))
        dws = np.linalg.norm(uv, axis=-1) - tdata[FV.AMB_WS]

        return dws, dwd        
