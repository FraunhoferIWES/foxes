import numpy as np

from foxes.core import Engine
import foxes.constants as FC


class DefaultEngine(Engine):
    """
    The case size dependent default engine.

    :group: engines

    """

    def submit(self, f, *args, **kwargs):
        """
        Submits a job to worker, obtaining a future

        Parameters
        ----------
        f: Callable
            The function f(*args, **kwargs) to be
            submitted
        args: tuple, optional
            Arguments for the function
        kwargs: dict, optional
            Arguments for the function

        Returns
        -------
        future: object
            The future object

        """
        self.finalize()

        with Engine.new(
            "process",
            n_procs=self.n_procs,
            chunk_size_states=self.chunk_size_states,
            chunk_size_points=self.chunk_size_points,
            verbosity=self.verbosity,
        ) as e:
            results = e.submit(f, *args, **kwargs)

        self.initialize()

        return results

    def await_result(self, future):
        """
        Waits for result from a future

        Parameters
        ----------
        future: object
            The future

        Returns
        -------
        result: object
            The calculation result

        """
        self.finalize()

        with Engine.new(
            "process",
            n_procs=self.n_procs,
            chunk_size_states=self.chunk_size_states,
            chunk_size_points=self.chunk_size_points,
            verbosity=self.verbosity,
        ) as e:
            results = e.await_result(future)

        self.initialize()

        return results

    def map(
        self,
        func,
        inputs,
        *args,
        **kwargs,
    ):
        """
        Runs a function on a list of files

        Parameters
        ----------
        func: Callable
            Function to be called on each file,
            func(input, *args, **kwargs) -> data
        inputs: array-like
            The input data list
        args: tuple, optional
            Arguments for func
        kwargs: dict, optional
            Keyword arguments for func

        Returns
        -------
        results: list
            The list of results

        """

        self.finalize()

        with Engine.new(
            "process",
            n_procs=self.n_procs,
            chunk_size_states=self.chunk_size_states,
            chunk_size_points=self.chunk_size_points,
            verbosity=self.verbosity,
        ) as e:
            results = e.map(func, inputs, *args, **kwargs)

        self.initialize()

        return results

    def run_calculation(
        self,
        algo,
        model,
        model_data,
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
        model_data: xarray.Dataset
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
            and algo.n_states * point_data.sizes[FC.TARGET] > 10000
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
