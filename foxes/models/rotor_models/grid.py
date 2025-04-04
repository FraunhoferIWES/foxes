import numpy as np

from foxes.core import RotorModel
from foxes.config import config


class GridRotor(RotorModel):
    """
    The weighted regular grid rotor model, composed maximally
    of n x n points, possibly kicking out the outside points.

    Attributes
    ----------
    n: int
        The number of points along one direction,
        maximal number of points is N = n * n
    reduce: bool
        Flag for reduction to points actually representing
        an area with overlap with the circle, recalculating
        the self.__weights accordingly
    nint: int
        Integration steps per element

    :group: models.rotor_models

    """

    def __init__(self, n, reduce=True, nint=200, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        n: int
            The number of points along one direction,
            maximal number of points is N = n * n
        reduce: bool
            Flag for reduction to points actually representing
            an area with overlap with the circle, recalculating
            the self.__weights accordingly
        nint: int
            Integration steps per element
        name: str, optional
            The model name
        kwargs: dict, optional
            Addition parameters for the base model

        """
        super().__init__(**kwargs)

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

        N = self.n * self.n
        delta = 2.0 / self.n
        x = [-1.0 + (i + 0.5) * delta for i in range(self.n)]
        x, y = np.meshgrid(x, x, indexing="ij")

        self.__dpoints = np.zeros([N, 3], dtype=config.dtype_double)
        self.__dpoints[:, 1] = x.reshape(N)
        self.__dpoints[:, 2] = y.reshape(N)

        if self.reduce:
            self.__weights = np.zeros((self.n, self.n), dtype=config.dtype_double)
            for i in range(0, self.n):
                for j in range(0, self.n):
                    d = delta / self.nint
                    hx = [
                        x[i, j] - delta / 2.0 + (k + 0.5) * d for k in range(self.nint)
                    ]
                    hy = [
                        y[i, j] - delta / 2.0 + (k + 0.5) * d for k in range(self.nint)
                    ]
                    pts = np.zeros((self.nint, self.nint, 2), dtype=config.dtype_double)
                    pts[:, :, 0], pts[:, :, 1] = np.meshgrid(hx, hy, indexing="ij")

                    d = np.linalg.norm(pts, axis=2)
                    self.__weights[i, j] = np.sum(d <= 1.0) / self.nint**2

            self.__weights = self.__weights.reshape(N)
            sel = self.__weights > 0.0
            self.__dpoints = self.__dpoints[sel]
            self.__weights = self.__weights[sel]
            self.__weights /= np.sum(self.__weights)

        else:
            self.__dpoints[:, 1] = x.reshape(N)
            self.__dpoints[:, 2] = y.reshape(N)
            self.__weights = np.ones(N, dtype=config.dtype_double) / N

    def n_rotor_points(self):
        """
        The number of rotor points

        Returns
        -------
        n_rpoints: int
            The number of rotor points

        """
        return len(self.__weights)

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
        return self.__dpoints

    def rotor_point_weights(self):
        """
        The weights of the rotor points

        Returns
        -------
        weights: numpy.ndarray
            The weights of the rotor points,
            add to one, shape: (n_rpoints,)

        """
        return self.__weights
