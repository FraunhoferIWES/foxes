
import numpy as np

from foxes.config import config
from foxes.core.wake_deflection import WakeDeflection
from foxes.core.wake_model import WakeK
from foxes.models.wake_models.wind.bastankhah16 import (
    Bastankhah2016Model,
    Bastankhah2016,
)
import foxes.constants as FC
import foxes.variables as FV


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
        alpha=0.58,
        beta=0.07,
        induction="Madsen",
        **wake_k,
    ):
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

        self.model = None
        self.alpha = alpha
        self.beta = beta
        self.induction = induction
        self.wake_k = None
        self._wake_k_pars = wake_k

        setattr(self, FV.YAWM, 0.0)

    def __repr__(self):
        s = f"{type(self).__name__}("
        s += self.wake_k.repr() if self.wake_k is not None else ""
        s += ")"
        return s

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self.wake_k, self.model]
    
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
        if not self.initialized:
            for w in algo.wake_models.values():
                if isinstance(w, Bastankhah2016):
                    if not w.initialized:
                        w.initialize(algo, verbosity, force)
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

        super().initialize(algo, verbosity, force)

    def _update_y(self, algo, mdata, fdata, tdata, downwind_index, x, y):
        """
        Helper function for y deflection
        """

        # get gamma:
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
        k = self.wake_k(
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
        self.model.calc_data(algo, mdata, fdata, tdata, downwind_index, x, gamma, k)

        # select targets:
        st_sel = self.model.get_data(Bastankhah2016Model.ST_SEL, mdata)
        if np.any(st_sel):
            # prepare:
            n_st_sel = np.sum(st_sel)
            ydef = np.zeros((n_st_sel,), dtype=config.dtype_double)

            # collect data:
            near = self.model.get_data(Bastankhah2016Model.NEAR, mdata)
            far = ~near

            # near wake:
            if np.any(near):
                # collect data:
                delta = self.model.get_data(Bastankhah2016Model.DELTA_NEAR, mdata)

                # set deflection:
                ydef[near] = delta

            # far wake:
            if np.any(far):
                # collect data:
                delta = self.model.get_data(Bastankhah2016Model.DELTA_FAR, mdata)

                # set deflection:
                ydef[far] = delta

            # apply deflection:
            y[st_sel] -= ydef

    def calc_deflection(
        self,
        algo, 
        mdata,
        fdata, 
        tdata, 
        downwind_index, 
        coos,
    ):
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
    