import numpy as np

from foxes.models.wake_models.top_hat import TopHatWakeModel
from foxes.utils.two_circles import calc_area
import foxes.variables as FV
import foxes.constants as FC

from .rotor_points import RotorPoints

class PartialTopHat(RotorPoints):
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
        if not isinstance(wmodel, TopHatWakeModel):
            raise TypeError(
                f"Partial wakes '{self.name}': Cannot be applied to wake model '{wmodel.name}', since not a TopHatWakeModel"
            )
        
        wcoos = algo.wake_frame.wake_coos_at_rotors(
            algo, mdata, fdata, pdata, downwind_index
        )
        x = wcoos[:, :, 0, 0]
        yz = wcoos[:, :, 0, 1:3]
        del wcoos

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
            R = np.linalg.norm(yz, axis=-1)
            del yz

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

            st_sel = sel0 & (wr > R - D / 2)
            if np.any(st_sel):
                x = x[st_sel]
                ct = ct[st_sel]
                wr = wr[st_sel]
                R = R[st_sel]
                D = D[st_sel]

                clw = wmodel.calc_centreline_wake_deltas(
                        algo, mdata, fdata, pdata, downwind_index,
                        st_sel, x, wr, ct)

                weights = calc_area(D / 2, wr, R) / (np.pi * (D / 2) ** 2)

                for v, d in clw.items():
                    try:
                        superp = wmodel.superp[v]
                    except KeyError:
                        raise KeyError(
                            f"Model '{self.name}': Missing wake superposition entry for variable '{v}' in wake model '{wmodel.name}', found {sorted(list(wmodel.superp.keys()))}"
                        )

                    wake_deltas[v] = superp.add_at_rotors(
                        algo, mdata, fdata, pdata, downwind_index, st_sel, 
                        v, wake_deltas[v], weights[:, None]*d[:, None])
