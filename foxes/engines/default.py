import numpy as np

from foxes.core import Engine
import foxes.constants as FC


class DefaultEngine(Engine):
    """
    The case size dependent default engine.

    :group: engines

    """

    def run_calculation(self, algo, *args, point_data=None, **kwargs):
        """
        Runs the model calculation

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm object
        args: tuple, optional
            Additional arguments for the calculation
        point_data: xarray.Dataset, optional
            The initial point data
        kwargs: dict, optional
            Additional arguments for the calculation

        Returns
        -------
        results: xarray.Dataset
            The model results

        """

        max_n = np.sqrt(self.n_procs) * (500 / algo.n_turbines) ** 1.5

        if (algo.n_states >= max_n) or (
            point_data is not None and point_data.sizes[FC.TARGET] >= 10000
        ):
            ename = "process"
        else:
            ename = "single"

        self.print(f"{type(self).__name__}: Selecting engine '{ename}'")

        self.finalize()

        with Engine.new(
            ename,
            n_procs=self.n_procs,
            chunk_size_states=self.chunk_size_states,
            chunk_size_points=self.chunk_size_points,
            verbosity=self.verbosity,
        ) as e:
            results = e.run_calculation(algo, *args, point_data=point_data, **kwargs)

        self.initialize()

        return results
