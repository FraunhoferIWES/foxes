import numpy as np
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.utils import write_nc

from .output import Output


class PointCalculator(Output):
    """
    Computes results at given points

    Attributes
    ----------
    algo: foxes.Algorithm
        The algorithm for point calculation
    farm_results: xarray.Dataset
        The farm results

    :group: output

    """

    def __init__(self, algo, farm_results, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        algo: foxes.Algorithm
            The algorithm for point calculation
        farm_results: xarray.Dataset
            The farm results
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(**kwargs)
        self.algo = algo
        self.farm_results = farm_results

    def calculate(
        self,
        points,
        *args,
        states_mean=False,
        weight_turbine=0,
        to_file=None,
        write_vars=None,
        write_pars={},
        **kwargs,
    ):
        """
        Calculate point results

        Parameters
        ----------
        points: numpy.ndarray
            The points, shape: (n_points, 3)
            or (n_states, n_points, 3)
        args: tuple, optional
            Additional arguments for algo.calc_points
        states_mean: bool
            Flag for taking the mean over states
        weight_turbine: int, optional
            Index of the turbine from which to take the weight
        to_file: str, optional
            Path to the output netCDF file
        write_vars: list of str
            The variables to be written to file, or None
            for all
        write_pars: dict, optional
            Additional parameters for write_nc
        kwargs: tuple, optional
            Additional arguments for algo.calc_points

        Returns
        -------
        point_results: xarray.Dataset
            The point results. The calculated variables have
            dimensions (state, point)

        """
        if points.shape[-1] == 3 and len(points.shape) == 3:
            pts = points
            p_has_s = True
        elif points.shape[-1] == 3 and len(points.shape) == 2:
            pts = np.zeros([self.algo.n_states] + list(points.shape), dtype=FC.DTYPE)
            pts[:] = points[None, :]
            p_has_s = False
        else:
            raise ValueError(
                f"Expecting point shape (n_states, n_points, 3) or (n_points, 3), got {points.shape}"
            )

        pres = self.algo.calc_points(self.farm_results, pts, *args, **kwargs)

        if states_mean:
            weights = self.farm_results[FV.WEIGHT].to_numpy()[:, weight_turbine]
            pres = Dataset(
                data_vars={
                    v: np.einsum("s,sp->p", weights, pres[v].to_numpy())
                    for v in pres.data_vars.keys()
                }
            )

        vrs = list(pres.data_vars.keys()) if write_vars is None else write_vars
        if to_file is not None:
            if states_mean:
                if p_has_s:
                    points = np.einsum("s,spd->pd", weights, points)
                dvars = {
                    "x": ((FC.POINT,), points[..., 0]),
                    "y": ((FC.POINT,), points[..., 1]),
                    "z": ((FC.POINT,), points[..., 2]),
                }
                dvars.update({v: ((FC.POINT,), pres[v].to_numpy()) for v in vrs})
                ds = Dataset(data_vars=dvars)
            else:
                if p_has_s:
                    dvars = {
                        "x": ((FC.STATE, FC.POINT), points[..., 0]),
                        "y": ((FC.STATE, FC.POINT), points[..., 1]),
                        "z": ((FC.STATE, FC.POINT), points[..., 2]),
                    }
                else:
                    dvars = {
                        "x": ((FC.POINT,), points[..., 0]),
                        "y": ((FC.POINT,), points[..., 1]),
                        "z": ((FC.POINT,), points[..., 2]),
                    }
                dvars.update(
                    {v: ((FC.STATE, FC.POINT), pres[v].to_numpy()) for v in vrs}
                )
                ds = Dataset(
                    coords={FC.STATE: pres[FC.STATE].to_numpy()}, data_vars=dvars
                )

            fpath = self.get_fpath(to_file)
            write_nc(ds, fpath, **write_pars)

        return pres
