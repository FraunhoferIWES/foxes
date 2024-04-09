import numpy as np

from foxes.core import PartialWakesModel, Data
from foxes.models.wake_models.top_hat import TopHatWakeModel
from foxes.utils.two_circles import calc_area
import foxes.variables as FV
import foxes.constants as FC


class PartialTopHat(PartialWakesModel):
    """
    Partial wakes for top-hat models.

    The wake effect is weighted by the overlap of
    the wake circle and the rotor disc circle.

    Attributes
    ----------
    rotor_model: foxes.core.RotorModel
        The rotor model, default is the one from the algorithm

    :group: models.partial_wakes

    """

    def __init__(self, rotor_model=None):
        """
        Constructor.

        Parameters
        ----------
        rotor_model: foxes.core.RotorModel, optional
            The rotor model, default is the one from the algorithm

        """
        super().__init__()
        self.rotor_model = rotor_model

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        if self.rotor_model is None:
            self.rotor_model = algo.rotor_model

        super().initialize(algo, verbosity)

        for w in self.wake_models:
            if not isinstance(w, TopHatWakeModel):
                raise TypeError(
                    f"Partial wakes '{self.name}': Cannot be applied to wake model '{w.name}', since not a TopHatWakeModel"
                )

        self.WCOOS_ID = self.var("WCOOS_ID")
        self.WCOOS_X = self.var("WCOOS_X")
        self.WCOOS_R = self.var("WCOOS_R")

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return super().sub_models() + [self.rotor_model]

    def get_wake_points(self, algo, mdata, fdata):
        """
        Get the wake calculation points.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data

        Returns
        -------
        rpoints: numpy.ndarray
            All rotor points, shape: (n_states, n_targets, n_rpoints, 3)

        """
        return fdata[FV.TXYH][:, :, None, :]

    def contribute_at_rotors(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        downwind_index,
        wake_deltas,
        wmodel,  
    ):
        """
        Modifies wake deltas at rotor points by 
        contributions from the specified wake source turbines.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data at rotor points
        downwind_index: int
            The index of the wake causing turbine
            in the downwnd order
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_rotors, n_rpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model

        """
        n_states = pdata.n_states
        n_rotors = pdata.n_rotors

        wcoos = algo.wake_frame.wake_coos_at_rotors(
            algo, mdata, fdata, pdata, downwind_index
        )
        x = wcoos[:, :, 0, 0]

        ct = self.get_data(
            FV.CT,
            FC.STATE_ROTOR,
            lookup="w",
            fdata=fdata,
            pdata=pdata,
            downwind_index=downwind_index,
            algo=algo,
        )

        sel0 = (ct > 0.0) & (x > 0.0)
        if np.any(sel0):
            R = np.linalg.norm(wcoos[:, :, :, 1:3], axis=-1)

            D = self.get_data(
                FV.D,
                FC.STATE_ROTOR,
                lookup="w",
                fdata=fdata,
                pdata=pdata,
                downwind_index=downwind_index,
                algo=algo,
            )

            wr = wmodel.calc_wake_radius(algo, mdata, fdata,
                            pdata, downwind_index, x, ct)
            TODO
            sel_sp = sel0 & (wr > R - D / 2)
            if np.any(sel_sp):
                hx = x[sel_sp]
                hct = ct[sel_sp]
                hwr = wr[sel_sp]

                clw = w.calc_centreline_wake_deltas(
                    algo,
                    mdata,
                    fdata,
                    pdata,
                    states_source_turbine,
                    sel_sp,
                    hx,
                    hwr,
                    hct,
                )
                del hx, hct

                hR = R[sel_sp]
                hD = D[sel_sp]
                weights = calc_area(hD / 2, hwr, hR) / (np.pi * (hD / 2) ** 2)
                del hD, hwr, hR

                for v, d in clw.items():
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
                        pdata,
                        states_source_turbine,
                        sel_sp,
                        v,
                        wake_deltas[v],
                        weights * d,
                    )

    def evaluate_results(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        wake_deltas,
        states_turbine,
        amb_res=None,
    ):
        """
        Updates the farm data according to the wake
        deltas.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
            Modified in-place by this function
        pdata: foxes.core.Data
            The evaluation point data
        wake_deltas: Any
            The wake deltas object at the selected downwind
            turbines
        states_turbine: numpy.ndarray of int
            For each state, the index of one turbine
            for which to evaluate the wake deltas.
            Shape: (n_states,)
        amb_res: dict, optional
            Ambient states results. Keys: var str, values:
            numpy.ndarray of shape (n_states, n_points)

        """
        weights = algo.rotor_model.from_data_or_store(FC.RWEIGHTS, algo, mdata)
        rpoints = algo.rotor_model.from_data_or_store(FC.RPOINTS, algo, mdata)
        n_states, n_turbines, n_rpoints, __ = rpoints.shape

        amb_res_in = amb_res is not None
        if not amb_res_in:
            amb_res = algo.rotor_model.from_data_or_store(
                FC.AMB_RPOINT_RESULTS, algo, mdata
            )

        wres = {}
        st_sel = (np.arange(n_states), states_turbine)
        for v, ares in amb_res.items():
            wres[v] = ares.reshape(n_states, n_turbines, n_rpoints)[st_sel]

        wdel = {}
        for v, d in wake_deltas.items():
            wdel[v] = d.reshape(n_states, n_turbines, 1)[st_sel]
        for w in self.wake_models:
            w.finalize_wake_deltas(algo, mdata, fdata, pdata, wres, wdel)

        for v in wres.keys():
            if v in wake_deltas:
                wres[v] += wdel[v]
                if amb_res_in:
                    amb_res[v][st_sel] = wres[v]
            wres[v] = wres[v][:, None]

        self.rotor_model.eval_rpoint_results(
            algo, mdata, fdata, wres, weights, states_turbine=states_turbine
        )
