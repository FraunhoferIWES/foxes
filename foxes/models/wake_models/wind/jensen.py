import numpy as np

from foxes.models.wake_models.top_hat import TopHatWakeModel
import foxes.variables as FV
import foxes.constants as FC


class JensenWake(TopHatWakeModel):
    """
    The Jensen wake model.

    Attributes
    ----------
    k: float, optional
        The wake growth parameter k. If not given here
        it will be searched in the farm data.
    k_var: str
        The variable name for k

    :group: models.wake_models.wind

    """

    def __init__(self, superposition, k=None, k_var=FV.K, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        superposition: str
            The wind deficit superposition
        k: float, optional
            The wake growth parameter k. If not given here
            it will be searched in the farm data.
        k_var: str
            The variable name for k
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(superpositions={FV.WS: superposition}, **kwargs)

        self.k_var = k_var
        setattr(self, k_var, k)

    def __repr__(self):
        k = getattr(self, self.k_var)
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        s = f"{type(self).__name__}"
        s += f"({self.superpositions[FV.WS]}, induction={iname}"
        if k is None:
            s += f", k_var={self.k_var}"
        else:
            s += f", {self.k_var}={k}"
        s += ")"
        return s

    def calc_wake_radius(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        x,
        ct,
    ):
        """
        Calculate the wake radius, depending on x only (not r).

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
            The index in the downwind order
        x: numpy.ndarray
            The x values, shape: (n_states, n_targets)
        ct: numpy.ndarray
            The ct values of the wake-causing turbines,
            shape: (n_states, n_targets)

        Returns
        -------
        wake_r: numpy.ndarray
            The wake radii, shape: (n_states, n_targets)

        """
        D = self.get_data(
            FV.D,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=False,
        )

        k = self.get_data(
            self.k_var,
            FC.STATE_TARGET,
            lookup="sw",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=False,
        )

        return D / 2 + k * x

    def calc_centreline(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        st_sel,
        x,
        wake_r,
        ct,
    ):
        """
        Calculate centre line results of wake deltas.

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
            The index in the downwind order
        st_sel: numpy.ndarray of bool
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)
        x: numpy.ndarray
            The x values, shape: (n_st_sel,)
        wake_r: numpy.ndarray
            The wake radii, shape: (n_st_sel,)
        ct: numpy.ndarray
            The ct values of the wake-causing turbines,
            shape: (n_st_sel,)

        Returns
        -------
        cl_del: dict
            The centre line wake deltas. Key: variable name str,
            varlue: numpy.ndarray, shape: (n_st_sel,)

        """
        R = (
            self.get_data(
                FV.D,
                FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=True,
            )[st_sel]
            / 2
        )

        twoa = 2 * self.induction.ct2a(ct)

        return {FV.WS: -((R / wake_r) ** 2) * twoa}
