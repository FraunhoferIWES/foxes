
import numpy as np

from foxes.config import config
from foxes.core.wake_deflection import WakeDeflection
import foxes.constants as FC
import foxes.variables as FV


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

            xmax = np.max(x)
            n_x = int(xmax/self.step_x)
            if xmax > n_x * self.step_x:
                n_x += 1

            delx = np.arange(n_x + 1) * self.step_x
            delx = np.minimum(delx[None, :], x[:, None])

            dx = delx[:, 1:] - delx[:, :-1]
            delx = delx[:, :-1]

            alpha = -np.cos(gamma[:, None])**2 * np.sin(gamma[:, None]) * ct[:, None]/2 / (
                1 + self.beta * delx / D[:, None]
            )**2
            del gamma, ct, delx, D

            y[sel] += np.sum(np.tan(alpha) * dx, axis=-1)
            coos[..., 1] = y[:, :, None]

        return coos

    