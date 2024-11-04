from copy import deepcopy

from foxes.utils import import_module

from .pool import PoolEngine


ray = None


def load_ray():
    """On-demand loading of the ray package"""
    global ray
    if ray is None:
        ray = import_module("ray", hint="pip install ray")


class RayEngine(PoolEngine):
    """
    The ray engine for foxes calculations.

    :group: engines

    """

    def _create_pool(self):
        """Creates the pool"""
        self.print(f"Initializing pool of {self.n_procs} ray workers")
        load_ray()
        ray.init(num_cpus=self.n_procs)

    def _submit(self, f, *args, **kwargs):
        """
        Submits to the pool

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

        @ray.remote
        def f_ray(*args, **kwargs):
            return f(*deepcopy(args), **deepcopy(kwargs))

        return f_ray.remote(*args, **kwargs)

    def _result(self, future):
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
        return ray.get(future)

    def _shutdown_pool(self):
        """Shuts down the pool"""
        self.print(f"Shutting down pool of {self.n_procs} ray workers")
        ray.shutdown()
