import dask
from abc import abstractmethod, ABCMeta
from copy import deepcopy
from dask.distributed import Client, LocalCluster
from dask.distributed import get_client
from dask.diagnostics import ProgressBar


class Runner(metaclass=ABCMeta):
    """
    Abstract base class for runners.

    :group: utils.runners

    """

    def __init__(self):
        self._initialized = False

    def initialize(self):
        """
        Initialize the runner
        """
        self._initialized = True

    @property
    def initialized(self):
        """
        Initialization flag

        Returns
        -------
        bool :
            Initialization flag

        """
        return self._initialized

    def __enter__(self):
        self.initialize()
        return self

    @abstractmethod
    def run(self, func, args=tuple(), kwargs={}):
        """
        Runs the given function.

        Parameters
        ----------
        func: Function
            The function to be run
        args: tuple
            The function arguments
        kwargs: dict
            The function keyword arguments

        Returns
        -------
        results: Any
            The functions return value

        """
        pass

    def finalize(self):
        """
        Finalize the runner
        """
        self._initialized = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()


class DefaultRunner(Runner):
    """
    Class for default function execution.

    :group: utils.runners

    """

    def run(self, func, args=tuple(), kwargs={}):
        """
        Runs the given function.

        Parameters
        ----------
        func: Function
            The function to be run
        args: tuple
            The function arguments
        kwargs: dict
            The function keyword arguments

        Returns
        -------
        results: Any
            The functions return value

        """
        return func(*args, **kwargs)


class DaskRunner(Runner):
    """
    Class for function execution via dask

    Attributes
    ----------
    scheduler: str, optional
        The dask scheduler choice
    progress_bar: bool
        Flag for showing progress bar
    cluster_args: dict, optional
        Explicit arguments for the cluster setup
    client_args: dict, optional
        Explicit arguments for the client setup
    verbosity: int
        The verbosity level, 0 = silent

    :group: utils.runners

    """

    def __init__(
        self,
        scheduler=None,
        n_workers=None,
        threads_per_worker=None,
        processes=True,
        cluster_args=None,
        client_args={},
        progress_bar=True,
        verbosity=1,
    ):
        """
        Constructor.

        Parameters
        ----------
        scheduler: str, optional
            The dask scheduler choice
        n_workers: int, optional
            The number of workers for parallel run
        threads_per_worker: int, optional
            The number of threads per worker for parallel run
        progress_bar: bool
            Flag for showing progress bar
        cluster_args: dict, optional
            Explicit arguments for the cluster setup
        client_args: dict, optional
            Explicit arguments for the client setup
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().__init__()

        self.scheduler = scheduler
        self.client_args = client_args
        self.progress_bar = progress_bar
        self.verbosity = verbosity

        if cluster_args is None:
            self.cluster_args = dict(
                n_workers=n_workers,
                processes=processes,
                threads_per_worker=threads_per_worker,
            )
        elif n_workers is not None or threads_per_worker is not None:
            raise KeyError(
                "Cannot handle 'n_workers', 'threads_per_worker' arguments if 'cluster_args' are provided"
            )
        else:
            self.cluster_args = cluster_args

        if scheduler is None and (
            n_workers is not None
            or threads_per_worker is not None
            or cluster_args is not None
        ):
            self.scheduler = "distributed"

    @classmethod
    def is_distributed(cls):
        try:
            get_client()
            return True
        except ValueError:
            return False

    def initialize(self):
        """
        Initialize the runner
        """
        if self.scheduler is not None:
            dask.config.set(scheduler=self.scheduler)

        if self.scheduler == "dthreads":
            self.print("Launching local dask cluster..")

            cargs = deepcopy(self.cluster_args)
            del cargs["processes"]
            self._client = Client(processes=False, **self.client_args, **cargs)

            self.print(f"Dashboard: {self._client.dashboard_link}\n")

        elif self.scheduler == "distributed":
            self.print("Launching local dask cluster..")

            self._cluster = LocalCluster(**self.cluster_args)
            self._client = Client(self._cluster, **self.client_args)

            self.print(self._cluster)
            self.print(f"Dashboard: {self._client.dashboard_link}\n")

        elif self.scheduler == "slurm":
            from dask_jobqueue import SLURMCluster

            self.print("Launching dask cluster on HPC using SLURM..")

            cargs = deepcopy(self.cluster_args)
            nodes = cargs.pop("nodes", 1)

            self._cluster = SLURMCluster(**cargs)
            self._cluster.scale(jobs=nodes)
            self._client = Client(self._cluster, **self.client_args)

            self.print(self._cluster)
            self.print(f"Dashboard: {self._client.dashboard_link}\n")

        super().initialize()

    def print(self, *args, **kwargs):
        """
        Prints if verbosity is not zero
        """
        if self.verbosity > 0:
            print(*args, **kwargs)

    def run(self, func, args=tuple(), kwargs={}):
        """
        Runs the given function.

        Parameters
        ----------
        func: Function
            The function to be run
        args: tuple
            The function arguments
        kwargs: dict
            The function keyword arguments

        Returns
        -------
        results: Any
            The functions return value

        """
        if self.progress_bar:
            with ProgressBar():
                results = func(*args, **kwargs)
        else:
            results = func(*args, **kwargs)

        return results

    def finalize(self):
        """
        Finallize the runner
        """
        if self.scheduler in ["distributed", "slurm"]:
            self.print("\n\nShutting down dask cluster")
            self._client.close()
            self._cluster.close()

        dask.config.refresh()

        super().finalize()
