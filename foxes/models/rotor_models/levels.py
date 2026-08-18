from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any

from foxes.core import RotorModel
from foxes.config import config
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.model import LoadedData


class LevelRotor(RotorModel):
    """
    The weighted regular rotor level model, composed of
    of n points between lower and upper blade tip.
    Calculates a height-dependent REWS

    Attributes
    ----------
    n
        The number of points along the vertical direction
    reduce
        Flag for calculating the weight of every element according
        to the rotor diameter at the respective height level
    nint
        Integration steps per element


    """

    def __init__(
        self, n: int, reduce: bool = True, nint: int = 200, **kwargs: Any
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        n
            The number of points along the vertical direction
        reduce
            Flag for calculating the weight of every element according
            to the rotor diameter at the respective height level
        nint
            Integration steps per element
        name
            The model name
        kwargs
            Addition parameters for the base model

        """
        super().__init__(**kwargs)

        self.n = n
        self.reduce = reduce
        self.nint = nint

    def __repr__(self) -> str:
        r = "" if self.reduce else ", reduce=False"
        return f"{type(self).__name__}(n={self.n}){r}"

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        loaded_data = super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

        delta = 2.0 / self.n
        y = [-1.0 + (i + 0.5) * delta for i in range(self.n)]
        x: np.ndarray = np.zeros(self.n, dtype=config.dtype_double)

        self.dpoints: np.ndarray = np.zeros([self.n, 3], dtype=config.dtype_double)
        self.dpoints[:, 1] = x
        self.dpoints[:, 2] = y

        if self.reduce:
            self.weights: np.ndarray = np.zeros((self.n), dtype=config.dtype_double)
            hx = np.linspace(1, -1, self.nint)

            for i in range(0, self.n):
                d = delta / self.nint
                hy = [y[i] - delta / 2.0 + (k + 0.5) * d for k in range(self.nint)]
                pts: np.ndarray = np.zeros(
                    (self.nint, self.nint, 2), dtype=config.dtype_double
                )
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
            self.weights = np.ones(self.n, dtype=config.dtype_double) / self.n

        self.weights = np.asarray(self.weights, dtype=config.dtype_double)

        return loaded_data

    def input_variables(self) -> list[str]:
        """
        The input variables which are required by the model.

        Returns
        -------
        input_vars
            The input variable names

        """
        return [FV.D, FV.TXYH, FV.YAW]

    def n_rotor_points(self) -> int:
        """
        The number of rotor points

        Returns
        -------
        n_rpoints
            The number of rotor points

        """
        return len(self.weights)

    def design_points(self) -> np.ndarray:
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
        dpoints
            The design points, shape: (n_points, 3)

        """
        return self.dpoints

    def rotor_point_weights(self) -> np.ndarray:
        """
        The weights of the rotor points

        Returns
        -------
        weights
            The weights of the rotor points,
            add to one, shape: (n_rpoints,)

        """
        return self.weights
