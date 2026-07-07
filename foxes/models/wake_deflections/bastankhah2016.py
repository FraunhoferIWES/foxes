from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any

from foxes.config import config
from foxes.core.wake_deflection import WakeDeflection
from foxes.core.wake_model import WakeK
from foxes.models.wake_models.wind.bastankhah16 import (
    Bastankhah2016Model,
    Bastankhah2016,
)
import foxes.constants as FC
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData


class Bastankhah2016Deflection(WakeDeflection):
    """
    Bend the wakes for yawed turbines, based on the
    Bastankhah 2016 wake model

    Notes
    -----
    Reference:
    "Experimental and theoretical study of wind turbine wakes in yawed conditions"
    Majid Bastankhah, Fernando Porté-Agel
    https://doi.org/10.1017/jfm.2016.595

    Attributes
    ----------
    model: Bastankhah2016Model
        The model for computing common data
    alpha: float
        model parameter used to determine onset of far wake region,
        if not found in wake model
    beta: float
        model parameter used to determine onset of far wake region,
        if not found in wake model
    wake_k: dict
        Parameters for the WakeK class, if not found in wake model
    induction: foxes.core.AxialInductionModel
        The induction model, if not found in wake model

    :group: models.wake_deflections

    """

    def __init__(
        self,
        alpha: float = 0.58,
        beta: float = 0.07,
        induction: Any = "Madsen",
        **wake_k: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        alpha: float
            model parameter used to determine onset of far wake region,
            if not found in wake model
        beta: float
            model parameter used to determine onset of far wake region,
            if not found in wake model
        induction: foxes.core.AxialInductionModel or str
            The induction model, if not found in wake model
        wake_k: dict, optional
            Parameters for the WakeK class, if not found in wake model

        """
        super().__init__()

        self.model: Bastankhah2016Model | None = None
        self.alpha = alpha
        self.beta = beta
        self.induction = induction
        self.wake_k: WakeK | None = None
        self._wake_k_pars = wake_k

        setattr(self, FV.YAWM, 0.0)

    def __repr__(self) -> str:
        s = f"{type(self).__name__}("
        s += self.wake_k.repr() if self.wake_k is not None else ""
        s += ")"
        return s

    def sub_models(self) -> list[Any]:
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        smdls: list[Any] = []
        if self.wake_k is not None:
            smdls.append(self.wake_k)
        if self.model is not None:
            smdls.append(self.model)
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
        if not self.initialized:
            for w in algo.wake_models.values():
                if isinstance(w, Bastankhah2016):
                    if not w.initialized:
                        w.initialize(
                            algo=algo,
                            loaded_data=loaded_data,
                            force=force,
                            verbosity=verbosity,
                        )
                    self.model = w.model
                    self.wake_k = w.wake_k
                    break
            if self.model is None:
                self.model = Bastankhah2016Model(
                    alpha=self.alpha, beta=self.beta, induction=self.induction
                )
            if self.wake_k is None:
                wake_k = WakeK(**self._wake_k_pars)
                if not wake_k.all_none:
                    self.wake_k = wake_k
                else:
                    for w in algo.wake_models.values():
                        if hasattr(w, "wake_k"):
                            self.wake_k = w.wake_k
                            break

        return super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

    def _update_y(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Helper function for y deflection
        """

        # get gamma:
        wake_k = self.wake_k
        model = self.model
        assert wake_k is not None and model is not None

        gamma = self.get_data(
            FV.YAWM,
            FC.STATE_TARGET,
            lookup="wfs",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            upcast=True,
            downwind_index=downwind_index,
            accept_nan=False,
        )
        gamma = gamma * np.pi / 180

        # get k:
        k = wake_k(
            FC.STATE_TARGET,
            lookup_ti="f",
            lookup_k="sf",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            upcast=True,
            downwind_index=downwind_index,
            accept_nan=False,
        )

        # run model calculation:
        model.calc_data(algo, mdata, fdata, tdata, downwind_index, x, gamma, k)

        # select targets:
        st_sel = model.get_cached_data(Bastankhah2016Model.ST_SEL, mdata)
        if np.any(st_sel):
            # prepare:
            n_st_sel = np.sum(st_sel)
            ydef = np.zeros((n_st_sel,), dtype=config.dtype_double)

            # collect data:
            near = model.get_cached_data(Bastankhah2016Model.NEAR, mdata)
            far = ~near

            # near wake:
            if np.any(near):
                # collect data:
                delta = model.get_cached_data(Bastankhah2016Model.DELTA_NEAR, mdata)

                # set deflection:
                ydef[near] = delta

            # far wake:
            if np.any(far):
                # collect data:
                delta = model.get_cached_data(Bastankhah2016Model.DELTA_FAR, mdata)

                # set deflection:
                ydef[far] = delta

            # apply deflection:
            y[st_sel] -= ydef

    def calc_deflection(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        coos: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the wake deflection.

        This function optionally adds FC.WDEFL_ROT_ANGLE or
        FC.WDEFL_DWS_FACTOR to the tdata.

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
        coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        Returns
        -------
        coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        """

        # take rotor average:
        xy = np.einsum("stpd,p->std", coos[..., :2], tdata[FC.TWEIGHTS])
        x = xy[:, :, 0]
        y = xy[:, :, 1]

        # apply deflection:
        self._update_y(algo, mdata, fdata, tdata, downwind_index, x, y)
        coos[..., 1] = y[:, :, None]

        return coos
