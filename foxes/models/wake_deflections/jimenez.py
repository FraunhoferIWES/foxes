
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
    rotate: bool
        If True, rotate local wind vector at evaluation points.
        If False, multiply wind speed with cos(angle) instead.
        If None, do not modify the wind vector, only the path.
    beta: float
        The beta coefficient of the Jimenez model
    step_x: float
        The x step in m for integration

    :group: models.wake_deflections

    """

    def __init__(self, rotate=True, beta=0.1, step_x=10.):
        """
        Constructor.
        
        Parameters
        ----------
        rotate: bool, optional
            If True, rotate local wind vector at evaluation points.
            If False, multiply wind speed with cos(angle) instead.
            If None, do not modify the wind vector, only the path.
        beta: float
            The beta coefficient of the Jimenez model
        step_x: float
            The x step in m for integration
        
        """
        super().__init__()
        self.rotate = rotate
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

            # calculate wind vector modification at evaluation points:
            if self.rotate is not None:

                # delta wd at evaluation points, if within wake radius:
                r2 = (y[sel, None]**2 + z[sel, None]**2) / D**2
                WD2 = (1 + self.beta * x[:, None] / D)**2
                delwd[sel] = np.where(r2 <= WD2 / 4, alpha0 / WD2, 0)

                if self.rotate:
                    tdata[FC.WDEFL_ROT_ANGLE] = np.rad2deg(delwd)
                else:
                    tdata[FC.WDEFL_DWS_FACTOR] = np.cos(delwd)

        return coos
    