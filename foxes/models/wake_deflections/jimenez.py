
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

    def calc_deflection(
        self,
        algo, 
        mdata,
        fdata, 
        tdata, 
        downwind_index, 
        wframe, 
        coos,
    ):
        """
        Calculates the wake deflection.

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
        coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        Returns
        -------
        coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)
        delta_wd_defl: numpy.ndarray or None
            The wind direction change at the target points 
            in radiants due to wake deflection, 
            shape: (n_states, n_targets, n_tpoints)

        """

        if FV.YAWM not in fdata:
            return coos, None

        # take rotor average:
        xyz = np.einsum("stpd,p->std", coos, tdata[FC.TWEIGHTS])
        x = xyz[:, :, 0]
        y = xyz[:, :, 1]
        z = xyz[:, :, 2]

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

        sel = (x > 1e-8) & (ct > 1e-8) & (np.abs(gamma) > 1e-8)
        delwd = np.zeros_like(coos[..., 0])
        n_sel = np.sum(sel)
        if n_sel > 0:

            # apply selection:
            gamma = np.deg2rad(gamma[sel])
            ct = ct[sel]
            x = x[sel]

            # get rotor diameter:
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
            )[:, None]

            # define x path:
            xmax = np.max(x)
            n_x = int(xmax/self.step_x)
            if xmax > n_x * self.step_x:
                n_x += 1
            delx = np.arange(n_x + 1) * self.step_x
            delx = np.minimum(delx[None, :], x[:, None])
            dx = delx[:, 1:] - delx[:, :-1]
            delx = delx[:, :-1]

            # integrate deflection of y along the x path:
            alpha0 = -np.cos(gamma[:, None])**2 * np.sin(gamma[:, None]) * ct[:, None]/2
            y[sel] += np.sum(np.tan(
                    alpha0 / (1 + self.beta * delx / D)**2
                ) * dx, axis=-1)
            coos[..., 1] = y[:, :, None]

            # delta wd at evaluation points, if within wake radius:
            r2 = (y[sel, None]**2 + z[sel, None]**2) / D**2
            WD2 = (1 + self.beta * x[:, None] / D)**2
            delwd[sel] = np.where(r2 <= WD2 / 4, np.rad2deg(alpha0 / WD2), 0)

        return coos, delwd
    