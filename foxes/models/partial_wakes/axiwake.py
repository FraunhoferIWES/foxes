import numpy as np

from foxes.models.wake_models.axisymmetric import AxisymmetricWakeModel
from foxes.utils.two_circles import calc_area
import foxes.variables as FV
import foxes.constants as FC

from .centre import PartialCentre


class PartialAxiwake(PartialCentre):
    """
    Partial wake calculation for axial wake models.

    The basic idea is that the x-dependent part of
    the wake model is evaluated only once, and the radial
    part then for `n` radii that cover the target rotor discs.

    The latter results are then weighted according to the overlap
    of radial wake circle area deltas and the target rotor disc area.

    Attributes
    ----------
    n: int
        The number of radial evaluation points

    :group: models.partial_wakes

    """

    def __init__(self, n=6):
        """
        Constructor.

        Parameters
        ----------
        n: int
            The number of radial evaluation points

        """
        super().__init__()
        self.n = n

    def __repr__(self):
        return f"{type(self).__name__}(n={self.n})"

    def check_wmodel(self, wmodel, error=True):
        """
        Checks the wake model type

        Parameters
        ----------
        wmodel: foxes.core.WakeModel
            The wake model to be tested
        error: bool
            Flag for raising TypeError

        Returns
        -------
        chk: bool
            True if wake model is compatible

        """
        if not isinstance(wmodel, AxisymmetricWakeModel):
            if error:
                raise TypeError(
                    f"Partial wakes '{self.name}': Cannot be applied to wake model '{wmodel.name}', since not an AxisymmetricWakeModel"
                )
            return False
        return True

    def contribute(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        wake_deltas,
        wmodel,
    ):
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

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
            in the downwnd order
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model

        """
        # check:
        self.check_wmodel(wmodel, error=True)

        # prepare:
        n_states = mdata.n_states
        n_targets = tdata.n_targets

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
        )

        # calc coordinates to rotor centres:
        wcoos = algo.wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)

        # prepare x and r coordinates:
        x = np.round(wcoos[..., 0, 0], 12)
        n = wcoos[..., 0, 1:3]
        R = np.linalg.norm(n, axis=-1)
        r = np.zeros((n_states, n_targets, self.n), dtype=FC.DTYPE)
        del wcoos

        # prepare circle section area calculation:
        A = np.zeros((n_states, n_targets, self.n), dtype=FC.DTYPE)
        weights = np.zeros_like(A)

        # get normalized 2D vector between rotor and wake centres:
        sel = R > 0
        if np.any(sel):
            n[sel] /= R[sel][:, None]
        if np.any(~sel):
            n[:, :, 0][~sel] = 1

        # case wake centre outside rotor disk:
        sel = (x > 0) & (R > D / 2)
        if np.any(sel):
            n_sel = np.sum(sel)
            Rsel = np.zeros((n_sel, self.n + 1), dtype=FC.DTYPE)
            Rsel[:] = R[sel][:, None]
            Dsel = D[sel][:, None]

            # equal delta R2:
            R1 = np.zeros((n_sel, self.n + 1), dtype=FC.DTYPE)
            R1[:] = Dsel / 2
            steps = np.linspace(0.0, 1.0, self.n + 1, endpoint=True) - 0.5
            R2 = np.zeros_like(R1)
            R2[:] = Rsel + Dsel * steps[None, :]
            r[sel] = 0.5 * (R2[:, 1:] + R2[:, :-1])

            hA = calc_area(R1, R2, Rsel)
            hA = hA[:, 1:] - hA[:, :-1] + 1e-15

            weights[sel] = hA / np.sum(hA, axis=-1)[:, None]
            del hA, Rsel, Dsel, R1, R2

        # case wake centre inside rotor disk:
        sel = (x > 0) & (R < D / 2)
        if np.any(sel):
            n_sel = np.sum(sel)
            Rsel = np.zeros((n_sel, self.n + 1), dtype=FC.DTYPE)
            Rsel[:] = R[sel][:, None]
            Dsel = D[sel][:, None]

            # equal delta R2:
            R1 = np.zeros((n_sel, self.n + 1), dtype=FC.DTYPE)
            R1[:, 1:] = Dsel / 2
            R2 = np.zeros_like(R1)
            # R2[:, 1:] = Rsel[:, :-1] + Dsel/2
            # R2[:]    *= np.linspace(0., 1, self.n + 1, endpoint=True)[None, :]
            R2[:, 1:] = (Rsel[:, :-1] + Dsel / 2) / (self.n - 0.5)
            R2[:, 1:] *= (
                0.5 + np.linspace(0.0, self.n - 1, self.n, endpoint=True)[None, :]
            )
            hr = 0.5 * (R2[:, 1:] + R2[:, :-1])
            hr[:, 0] = 0.0
            r[sel] = hr

            hA = calc_area(R1, R2, Rsel)
            hA = hA[:, 1:] - hA[:, :-1]
            weights[sel] = hA / np.sum(hA, axis=-1)[:, None]
            del hA, hr, Rsel, Dsel, R1, R2

        # evaluate wake model:
        wdeltas, st_sel = wmodel.calc_wakes_x_r(
            algo, mdata, fdata, tdata, downwind_index, x, r
        )

        for v, wdel in wdeltas.items():
            d = np.einsum("sn,sn->s", wdel, weights[st_sel])

            try:
                superp = wmodel.superp[v]
            except KeyError:
                raise KeyError(
                    f"Model '{self.name}': Missing wake superposition entry for variable '{v}' in wake model '{wmodel.name}', found {sorted(list(wmodel.superp.keys()))}"
                )

            wake_deltas[v] = superp.add_wake(
                algo,
                mdata,
                fdata,
                tdata,
                downwind_index,
                st_sel,
                v,
                wake_deltas[v],
                d[:, None],
            )
