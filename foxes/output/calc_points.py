import numpy as np

import foxes.constants as FC

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
            write_csv=None,
            write_vars=None,
            round="auto",
            **kwargs
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
        write_csv: str, optional
            Path to the output csv file
        write_vars: list of str
            The variables to be written to file, or None
            for all
        kwargs: tuple, optional
            Additional arguments for algo.calc_points

        Returns
        -------
        point_results: xarray.Dataset
            The point results. The calculated variables have
            dimensions (state, point)

        """
        if len(points.shape) == 3:
            pts = points
        elif len(points.shape) == 2:
            pts = np.zeros([self.algo.n_states] + list(points.shape), dtype=FC.DTYPE)
            pts[:] = points[None, :]
        
        pres = self.algo.calc_points(self.farm_results, pts, *args, **kwargs)

        vrs = list(pres.data_vars.keys()) if write_vars is None else write_vars
        if write_csv is not None:
            fpath = self.get_fpath(write_csv)


        return pres
    