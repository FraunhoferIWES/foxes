import numpy as np

from foxes.core import WakeFrame, WakeK, TData
from foxes.models.wake_models.wind.bastankhah16 import (
    Bastankhah2016Model,
    Bastankhah2016,
)
import foxes.variables as FV
import foxes.constants as FC
from .rotor_wd import RotorWD


class YawedWakes(WakeFrame):
    """
    Bend the wakes for yawed turbines, based on the
    Bastankhah 2016 wake model

    Notes
    -----
    Reference:
    "Experimental and theoretical study of wind turbine wakes in yawed conditions"
    Majid Bastankhah, Fernando Porté-Agel
    https://doi.org/10.1017/jfm.2016.595

    Attributes
    ----------
    model: Bastankhah2016Model
        The model for computing common data
    model_pars: dict
        Model parameters
    YAWM: float
        The yaw misalignment YAWM. If not given here
        it will be searched in the farm data.
    base_frame: foxes.core.WakeFrame
        The wake frame from which to start

    :group: models.wake_frames

    """

    def __init__(
        self,
        base_frame=RotorWD(),
        alpha=0.58,
        beta=0.07,
        induction="Madsen",
        **wake_k,
    ):
        """
        Constructor.

        Parameters
        ----------
        base_frame: foxes.core.WakeFrame
            The wake frame from which to start
        alpha: float
            model parameter used to determine onset of far wake region,
            if not found in wake model
        beta: float
            model parameter used to determine onset of far wake region,
            if not found in wake model
        induction: foxes.core.AxialInductionModel or str
            The induction model, if not found in wake model
        wake_k: dict, optional
            Parameters for the WakeK class, if not found in wake model

        """
        super().__init__()

        self.base_frame = base_frame
        self.model = None
        self.alpha = alpha
        self.beta = beta
        self.induction = induction
        self.wake_k = None
        self._wake_k_pars = wake_k

        setattr(self, FV.YAWM, 0.0)

    def __repr__(self):
        s = f"{type(self).__name__}("
        s += self.wake_k.repr() if self.wake_k is not None else ""
        s += ")"
        return s

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self.wake_k, self.base_frame, self.model]

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        if not self.initialized:
            for w in algo.wake_models.values():
                if isinstance(w, Bastankhah2016):
                    if not w.initialized:
                        w.initialize(algo, verbosity, force)
                    self.model = w.model
                    self.wake_k = w.wake_k
                    break
            if self.model is None:
                self.model = Bastankhah2016Model(
                    alpha=self.alpha, beta=self.beta, induction=self.induction
                )
            if self.wake_k is None:
                wake_k = WakeK(**self._wake_k_pars)
                if not wake_k.all_none:
                    self.wake_k = wake_k
                else:
                    for w in algo.wake_models.values():
                        if hasattr(w, "wake_k"):
                            self.wake_k = w.wake_k
                            break

        super().initialize(algo, verbosity, force)

    def calc_order(self, algo, mdata, fdata):
        """ "
        Calculates the order of turbine evaluation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data

        Returns
        -------
        order: numpy.ndarray
            The turbine order, shape: (n_states, n_turbines)

        """
        return self.base_frame.calc_order(algo, mdata, fdata)

    def _update_y(self, algo, mdata, fdata, tdata, downwind_index, x, y):
        """
        Helper function for y deflection
        """

        # get gamma:
        gamma = self.get_data(
            FV.YAWM,
            FC.STATE_TARGET,
            lookup="wfs",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            upcast=True,
            downwind_index=downwind_index,
            accept_nan=False,
        )
        gamma *= np.pi / 180

        # get k:
        k = self.wake_k(
            FC.STATE_TARGET,
            lookup_ti="f",
            lookup_k="sf",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            upcast=True,
            downwind_index=downwind_index,
            accept_nan=False,
        )

        # run model calculation:
        self.model.calc_data(algo, mdata, fdata, tdata, downwind_index, x, gamma, k)

        # select targets:
        st_sel = self.model.get_data(Bastankhah2016Model.ST_SEL, mdata)
        if np.any(st_sel):
            # prepare:
            n_st_sel = np.sum(st_sel)
            ydef = np.zeros((n_st_sel,), dtype=FC.DTYPE)

            # collect data:
            near = self.model.get_data(Bastankhah2016Model.NEAR, mdata)
            far = ~near

            # near wake:
            if np.any(near):
                # collect data:
                delta = self.model.get_data(Bastankhah2016Model.DELTA_NEAR, mdata)

                # set deflection:
                ydef[near] = delta

            # far wake:
            if np.any(far):
                # collect data:
                delta = self.model.get_data(Bastankhah2016Model.DELTA_FAR, mdata)

                # set deflection:
                ydef[far] = delta

            # apply deflection:
            y[st_sel] -= ydef

    def get_wake_coos(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
    ):
        """
        Calculate wake coordinates of rotor points.

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

        Returns
        -------
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        """
        # get unyawed results:
        xyz = self.base_frame.get_wake_coos(
            algo,
            mdata,
            fdata,
            tdata,
            downwind_index,
        )

        # take rotor average:
        xy = np.einsum("stpd,p->std", xyz[..., :2], tdata[FC.TWEIGHTS])
        x = xy[:, :, 0]
        y = xy[:, :, 1]

        # apply deflection:
        self._update_y(algo, mdata, fdata, tdata, downwind_index, x, y)
        xyz[..., 1] = y[:, :, None]

        return xyz

    def get_centreline_points(self, algo, mdata, fdata, downwind_index, x):
        """
        Gets the points along the centreline for given
        values of x.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        downwind_index: int
            The index in the downwind order
        x: numpy.ndarray
            The wake frame x coordinates, shape: (n_states, n_points)

        Returns
        -------
        points: numpy.ndarray
            The centreline points, shape: (n_states, n_points, 3)

        """
        points = self.base_frame.get_centreline_points(
            algo, mdata, fdata, downwind_index, x
        )
        tdata = TData.from_points(points)

        nx = np.zeros_like(points)
        nx[:, 0] = points[:, 1] - points[:, 0]
        nx[:, -1] = points[:, -1] - points[:, -2]
        nx[:, 1:-1] = 0.5 * (points[:, 1:-1] - points[:, :-2]) + 0.5 * (
            points[:, 2:] - points[:, 1:-1]
        )
        nx /= np.linalg.norm(nx, axis=-1)[:, :, None]

        nz = np.zeros_like(nx)
        nz[:, :, 2] = 1
        ny = np.cross(nz, nx, axis=-1)
        del nx, nz

        y = np.zeros_like(x)
        self._update_y(algo, mdata, fdata, tdata, downwind_index, x, y)

        points += y[:, :, None] * ny

        return points
