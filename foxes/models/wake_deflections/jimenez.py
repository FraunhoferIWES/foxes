
import numpy as np

from foxes.config import config
from foxes.core.wake_deflection import WakeDeflection
import foxes.constants as FC
import foxes.variables as FV
from foxes.models.wake_frames import (
    RotorWD,
    FarmOrder,
)

class JimenezDeflection(WakeDeflection):
    """
    Yawed rotor wake defection according to the Jimenez model

    Notes
    -----
    Reference:
    "Experimental and theoretical study of wind turbine wakes in yawed conditions"
    Majid Bastankhah, Fernando Porté-Agel
    https://doi.org/10.1017/jfm.2016.595

    Attributes
    ----------
    beta: float
        The beta coefficient of the Jimenez model
    step_x: float
        The x step in m for integration

    :group: models.wake_deflections

    """

    def __init__(self, beta=0.1, step_x=10.):
        """
        Constructor.
        
        Parameters
        ----------
        beta: float
            The beta coefficient of the Jimenez model
        step_x: float
            The x step in m for integration
        
        """
        super().__init__()
        self.beta = beta
        self.step_x = step_x

    def update_coos(
        self,
        algo, 
        mdata,
        fdata, 
        tdata, 
        downwind_index, 
        wframe, 
        wmodel, 
        coos,
    ):
        """
        Updates the wake coordinates

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
        wframe: foxes.core.WakeFrame
            The wake frame
        wmodel: foxes.core.WakeModel
            The wake model
        coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        """
        if FV.YAWM not in fdata:
            return coos

        if isinstance(wframe, (RotorWD, FarmOrder)):

            # take rotor average:
            xy = np.einsum("stpd,p->std", coos[..., :2], tdata[FC.TWEIGHTS])
            x = xy[:, :, 0]
            y = xy[:, :, 1]

            # get ct:
            ct = self.get_data(
                FV.CT,
                FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=True,
            )

            # get gamma:
            gamma = self.get_data(
                FV.YAWM,
                FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=True,
            )

            sel = (x > 0) & (ct > 1e-8) & (np.abs(gamma) > 1e-8)
            n_sel = np.sum(sel)
            if n_sel > 0:

                # apply selection:
                gamma = gamma[sel] * np.pi / 180
                ct = ct[sel]
                x = x[sel]

                # get D:
                D = self.get_data(
                    FV.D,
                    FC.STATE_TARGET,
                    lookup="w",
                    algo=algo,
                    fdata=fdata,
                    tdata=tdata,
                    downwind_index=downwind_index,
                    upcast=True,
                    selection=sel,
                )

                alpha0 = -np.cos(gamma)**2 * np.sin(gamma) * ct/2
                delx = np.zeros_like(x)
                dely = np.zeros_like(x)
                sel2 = (delx + self.step_x <= x)
                while np.any(sel2):

                    dely[sel2] += np.tan(
                        alpha0[sel2] / (
                            1 + self.beta * (x[sel2] + delx[sel2]) / D[sel2]
                        )**2
                    ) * self.step_x

                    delx[sel2] += self.step_x
                    sel2 = (delx + self.step_x <= x)

                y[sel] += dely
                coos[..., 1] = y[:, :, None]

            return coos
            
        else:
            return super().update_coos(algo, mdata, fdata, tdata,
                                downwind_index, wframe, wmodel, coos)

    