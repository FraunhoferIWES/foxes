import numpy as np

from foxes.core import Engine
import foxes.constants as FC


class DefaultEngine(Engine):
    """
    The case size dependent default engine.

    :group: engines

    """

    def run_calculation(
        self,
        algo,
        model,
        model_data=None,
        farm_data=None,
        point_data=None,
        **kwargs,
    ):
        """
        Runs the model calculation

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm object
        model: foxes.core.DataCalcModel
            The model that whose calculate function
            should be run
        model_data: xarray.Dataset, optional
            The initial model data
        farm_data: xarray.Dataset, optional
            The initial farm data
        point_data: xarray.Dataset, optional
            The initial point data

        Returns
        -------
        results: xarray.Dataset
            The model results

        """
        max_n = np.sqrt(self.n_procs) * (500 / algo.n_turbines) ** 1.5

        if (algo.n_states >= max_n) or (
            point_data is not None
            and self.chunk_size_points is not None
            and point_data.sizes[FC.TARGET] > self.chunk_size_points
        ):
            ename = "process"
        else:
            ename = "single"

        self.print(f"{type(self).__name__}: Selecting engine '{ename}'", level=1)

        self.finalize()

        with Engine.new(
            ename,
            n_procs=self.n_procs,
            chunk_size_states=self.chunk_size_states,
            chunk_size_points=self.chunk_size_points,
            verbosity=self.verbosity,
        ) as e:
            results = e.run_calculation(
                algo, model, model_data, farm_data, point_data=point_data, **kwargs
            )

        self.initialize()

        return results
