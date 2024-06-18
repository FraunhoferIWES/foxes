import numpy as np

from foxes.core import RotorModel
import foxes.constants as FC


class LevelRotor(RotorModel):
    """
    The weighted regular rotor level model, composed of
    of n points between lower and upper blade tip.
    Calculates a height-dependent REWS

    Attributes
    ----------
    n: int
        The number of points along the vertical direction
    reduce: bool
        Flag for calculating the weight of every element according
        to the rotor diameter at the respective height level
    nint: int
        Integration steps per element

    :group: models.rotor_models

    """

    def __init__(self, n, calc_vars, reduce=True, nint=200):
        """
        Constructor.

        Parameters
        ----------
        n: int
            The number of points along the vertical direction
        calc_vars: list of str
            The variables that are calculated by the model
            (Their ambients are added automatically)
        reduce: bool
            Flag for calculating the weight of every element according
            to the rotor diameter at the respective height level
        nint: int
            Integration steps per element
        name: str, optional
            The model name

        """
        super().__init__(calc_vars)

        self.n = n
        self.reduce = reduce
        self.nint = nint

    def __repr__(self):
        r = "" if self.reduce else ", reduce=False"
        return f"{type(self).__name__}(n={self.n}){r}"

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
        super().initialize(algo, verbosity)

        delta = 2.0 / self.n
        y = [-1.0 + (i + 0.5) * delta for i in range(self.n)]
        x = np.zeros(self.n, dtype=FC.DTYPE)

        self.dpoints = np.zeros([self.n, 3], dtype=FC.DTYPE)
        self.dpoints[:, 1] = x
        self.dpoints[:, 2] = y

        if self.reduce:
            self.weights = np.zeros((self.n), dtype=FC.DTYPE)
            hx = np.linspace(1, -1, self.nint)

            for i in range(0, self.n):
                d = delta / self.nint
                hy = [y[i] - delta / 2.0 + (k + 0.5) * d for k in range(self.nint)]
                pts = np.zeros((self.nint, self.nint, 2), dtype=FC.DTYPE)
                pts[:, :, 0], pts[:, :, 1] = np.meshgrid(hx, hy, indexing="ij")

                d = np.linalg.norm(pts, axis=2)
                self.weights[i] = np.sum(d <= 1.0) / self.nint**2

            sel = self.weights > 0.0
            self.dpoints = self.dpoints[sel]
            self.weights = self.weights[sel]
            self.weights /= np.sum(self.weights)

        else:
            self.dpoints[:, 1] = x
            self.dpoints[:, 2] = y
            self.weights = np.ones(self.n, dtype=FC.DTYPE) / self.n

    def n_rotor_points(self):
        """
        The number of rotor points

        Returns
        -------
        n_rpoints: int
            The number of rotor points

        """
        return len(self.weights)

    def design_points(self):
        """
        The rotor model design points.

        Design points are formulated in rotor plane
        (x,y,z)-coordinates in rotor frame, such that
        - (0,0,0) is the centre point,
        - (1,0,0) is the point radius * n_rotor_axis
        - (0,1,0) is the point radius * n_rotor_side
        - (0,0,1) is the point radius * n_rotor_up

        Returns
        -------
        dpoints: numpy.ndarray
            The design points, shape: (n_points, 3)

        """
        return self.dpoints

    def rotor_point_weights(self):
        """
        The weights of the rotor points

        Returns
        -------
        weights: numpy.ndarray
            The weights of the rotor points,
            add to one, shape: (n_rpoints,)

        """
        return self.weights
