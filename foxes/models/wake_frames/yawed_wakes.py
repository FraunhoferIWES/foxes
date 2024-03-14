import numpy as np

from foxes.core import WakeFrame
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
    K: float
        The wake growth parameter k. If not given here
        it will be searched in the farm data.
    YAWM: float
        The yaw misalignment YAWM. If not given here
        it will be searched in the farm data.
    base_frame: foxes.core.WakeFrame
        The wake frame from which to start
    k_var: str
        The variable name for k

    :group: models.wake_frames

    """

    def __init__(
        self,
        k=None,
        base_frame=RotorWD(),
        k_var=FV.K,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        k: float, optional
            The wake growth parameter k. If not given here
            it will be searched in the farm data, by default None
        base_frame: foxes.core.WakeFrame
            The wake frame from which to start
        k_var: str
            The variable name for k
        kwargs: dict, optional
            Additional parameters for the Bastankhah2016Model,
            if not found in wake model

        """
        super().__init__()

        self.base_frame = base_frame
        self.model = None
        self.model_pars = kwargs
        self.k_var = k_var

        setattr(self, k_var, k)
        setattr(self, FV.YAWM, 0.0)

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self.base_frame, self.model]

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
            for w in algo.wake_models:
                if isinstance(w, Bastankhah2016):
                    if not w.initialized:
                        w.initialize(algo, verbosity, force)
                    self.model = w.model
                    if w.k_var == self.k_var:
                        setattr(self, self.k_var, getattr(w, self.k_var))
            if self.model is None:
                self.model = Bastankhah2016Model(**self.model_pars)
            if self.k is None:
                for w in algo.wake_models:
                    if hasattr(w, "k_var") and w.k_var == self.k_var:
                        setattr(self, self.k_var, getattr(w, self.k_var))
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
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data

        Returns
        -------
        order: numpy.ndarray
            The turbine order, shape: (n_states, n_turbines)

        """
        return self.base_frame.calc_order(algo, mdata, fdata)

    def _update_y(self, algo, mdata, fdata, pdata, states_source_turbine, x, y):
        """
        Helper function for y deflection
        """

        # get gamma:
        gamma = self.get_data(
            FV.YAWM,
            FC.STATE_POINT,
            lookup="wfs",
            algo=algo,
            fdata=fdata,
            pdata=pdata,
            upcast=True,
            states_source_turbine=states_source_turbine,
        )
        gamma *= np.pi / 180

        # get k:
        k = self.get_data(
            self.k_var,
            FC.STATE_POINT,
            lookup="sf",
            algo=algo,
            fdata=fdata,
            pdata=pdata,
            upcast=True,
            states_source_turbine=states_source_turbine,
        )

        # run model calculation:
        self.model.calc_data(
            algo, mdata, fdata, pdata, states_source_turbine, x, gamma, k
        )

        # select targets:
        sp_sel = self.model.get_data(Bastankhah2016Model.SP_SEL, mdata)
        if np.any(sp_sel):
            # prepare:
            n_sp_sel = np.sum(sp_sel)
            ydef = np.zeros((n_sp_sel,), dtype=FC.DTYPE)

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
            y[sp_sel] -= ydef

    def get_wake_coos(self, algo, mdata, fdata, pdata, states_source_turbine):
        """
        Calculate wake coordinates.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)

        Returns
        -------
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_points, 3)

        """

        # get unyawed results:
        xyz = self.base_frame.get_wake_coos(
            algo, mdata, fdata, pdata, states_source_turbine
        )
        x = xyz[:, :, 0]
        y = xyz[:, :, 1]

        # apply deflection:
        self._update_y(algo, mdata, fdata, pdata, states_source_turbine, x, y)

        return xyz

    def get_centreline_points(self, algo, mdata, fdata, states_source_turbine, x):
        """
        Gets the points along the centreline for given
        values of x.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        x: numpy.ndarray
            The wake frame x coordinates, shape: (n_states, n_points)

        Returns
        -------
        points: numpy.ndarray
            The centreline points, shape: (n_states, n_points, 3)

        """
        points = self.base_frame.get_centreline_points(
            algo, mdata, fdata, states_source_turbine, x
        )

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
        self._update_y(algo, mdata, fdata, None, states_source_turbine, x, y)

        points += y[:, :, None] * ny

        return points
