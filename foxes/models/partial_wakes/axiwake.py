import numpy as np

from foxes.core import PartialWakesModel
from foxes.models.wake_models.axisymmetric import AxisymmetricWakeModel
from foxes.utils.two_circles import calc_area
import foxes.variables as FV
import foxes.constants as FC


class PartialAxiwake(PartialWakesModel):
    """
    Partial wake calculation for axial wake models.

    The basic idea is that the x-dependent part of
    the wake model is evaluated only once, and the radial
    part then for `n` radii that cover the target rotor discs.

    The latter results are then weighted according to the overlap
    of radial wake circle area deltas and the target rotor disc area.

    Parameters
    ----------
    n : int
        The number of radial evaluation points
    wake_models : list of foxes.core.WakeModel, optional
        The wake models, default are the ones from the algorithm
    wake_frame : foxes.core.WakeFrame, optional
        The wake frame, default is the one from the algorithm
    rotor_model : foxes.core.RotorModel, optional
        The rotor model, default is the one from the algorithm

    Attributes
    ----------
    n : int
        The number of radial evaluation points
    rotor_model : foxes.core.RotorModel
        The rotor model, default is the one from the algorithm

    """

    def __init__(self, n, wake_models=None, wake_frame=None, rotor_model=None):
        super().__init__(wake_models, wake_frame)

        self.n = n
        self.rotor_model = rotor_model

    def __repr__(self):
        return super().__repr__() + f"(n={self.n})"

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level

        """
        super().initialize(algo, verbosity=verbosity)

        if self.rotor_model is None:
            self.rotor_model = algo.rotor_model
        if not self.rotor_model.initialized:
            self.rotor_model.initialize(algo, verbosity=verbosity)

        for w in self.wake_models:
            if not isinstance(w, AxisymmetricWakeModel):
                raise TypeError(
                    f"Partial wakes '{self.name}': Cannot be applied to wake model '{w.name}', since not an AxisymmetricWakeModel"
                )

    def new_wake_deltas(self, algo, mdata, fdata):
        """
        Creates new initial wake deltas, filled
        with zeros.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data

        Returns
        -------
        wake_deltas : dict
            Keys: Variable name str, values: any

        """
        n_points = fdata.n_turbines
        wake_deltas = {}
        for w in self.wake_models:
            w.init_wake_deltas(algo, mdata, fdata, n_points, wake_deltas)

        return wake_deltas

    def contribute_to_wake_deltas(
        self, algo, mdata, fdata, states_source_turbine, wake_deltas
    ):
        """
        Modifies wake deltas by contributions from the
        specified wake source turbines.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray of int
            For each state, one turbine index corresponding
            to the wake causing turbine. Shape: (n_states,)
        wake_deltas : Any
            The wake deltas object created by the
            `new_wake_deltas` function

        """
        # prepare:
        n_states = mdata.n_states
        n_turbines = algo.n_turbines
        D = fdata[FV.D]

        # calc coordinates to rotor centres:
        wcoos = self.wake_frame.get_wake_coos(
            algo, mdata, fdata, states_source_turbine, fdata[FV.TXYH]
        )

        # prepare x and r coordinates:
        x = wcoos[:, :, 0]
        n = wcoos[:, :, 1:3]
        R = np.linalg.norm(n, axis=-1)
        r = np.zeros((n_states, n_turbines, self.n), dtype=FC.DTYPE)
        del wcoos

        # prepare circle section area calculation:
        A = np.zeros((n_states, n_turbines, self.n), dtype=FC.DTYPE)
        weights = np.zeros_like(A)

        # get normalized 2D vector between rotor and wake centres:
        sel = R > 0.0
        if np.any(sel):
            n[sel] /= R[sel][:, None]
        if np.any(~sel):
            n[:, :, 0][~sel] = 1

        # case wake centre outside rotor disk:
        sel = (x > 1e-5) & (R > D / 2)
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
            hA = hA[:, 1:] - hA[:, :-1]
            weights[sel] = hA / np.sum(hA, axis=-1)[:, None]
            del hA, Rsel, Dsel, R1, R2

        # case wake centre inside rotor disk:
        sel = (x > 1e-5) & (R < D / 2)
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

        # evaluate wake models:
        for w in self.wake_models:

            wdeltas, sp_sel = w.calc_wakes_spsel_x_r(
                algo, mdata, fdata, states_source_turbine, x, r
            )

            for v, wdel in wdeltas.items():

                d = np.einsum("ps,ps->p", wdel, weights[sp_sel])

                try:
                    superp = w.superp[v]
                except KeyError:
                    raise KeyError(
                        f"Model '{self.name}': Missing wake superposition entry for variable '{v}' in wake model '{w.name}', found {sorted(list(w.superp.keys()))}"
                    )

                wake_deltas[v] = superp.calc_wakes_plus_wake(
                    algo,
                    mdata,
                    fdata,
                    states_source_turbine,
                    sp_sel,
                    v,
                    wake_deltas[v],
                    d,
                )

    def evaluate_results(
        self, algo, mdata, fdata, wake_deltas, states_turbine, update_amb_res=False
    ):
        """
        Updates the farm data according to the wake
        deltas.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
            Modified in-place by this function
        wake_deltas : Any
            The wake deltas object, created by the
            `new_wake_deltas` function and filled
            by `contribute_to_wake_deltas`
        states_turbine : numpy.ndarray of int
            For each state, the index of one turbine
            for which to evaluate the wake deltas.
            Shape: (n_states,)
        update_amb_res : bool
            Flag for updating ambient results

        """
        weights = self.get_data(FV.RWEIGHTS, mdata)
        amb_res = self.get_data(FV.AMB_RPOINT_RESULTS, mdata)
        rpoints = self.get_data(FV.RPOINTS, mdata)
        n_states, n_turbines, n_rpoints, __ = rpoints.shape

        wres = {}
        st_sel = (np.arange(n_states), states_turbine)
        for v, ares in amb_res.items():
            wres[v] = ares.reshape(n_states, n_turbines, n_rpoints)[st_sel]
        del amb_res

        wdel = {}
        for v, d in wake_deltas.items():
            wdel[v] = d.reshape(n_states, n_turbines, 1)[st_sel]
        for w in self.wake_models:
            w.finalize_wake_deltas(algo, mdata, fdata, wres, wdel)

        for v in wres.keys():
            if v in wake_deltas:
                wres[v] += wdel[v]
                if update_amb_res:
                    mdata[FV.AMB_RPOINT_RESULTS][v][st_sel] = wres[v]
            wres[v] = wres[v][:, None]

        self.rotor_model.eval_rpoint_results(
            algo, mdata, fdata, wres, weights, states_turbine=states_turbine
        )
