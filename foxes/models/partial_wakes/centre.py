import foxes.constants as FC
import foxes.variables as FV

from .rotor_points import RotorPoints


class PartialCentre(RotorPoints):
    """
    Partial wakes calculated only at the 
    rotor centre point.

    :group: models.partial_wakes

    """

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
            All rotor points, shape: (n_states, n_turbines, n_rpoints, 3)

        """
        return fdata[FV.TXYH][:, :, None]

    def n_wpoints(self, algo):
        """
        The number of evaluation points per rotor

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        n: int
            The number of evaluation points per rotor
        
        """
        return 1
    