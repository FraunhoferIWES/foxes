import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

from foxes.core import Runner

class DaskRunner(Runner):
    """
    Class for function execution via dask

    Parameters
    ----------
    scheduler : str, optional
        The dask scheduler choice
    n_workers : int, optional
        The number of workers for parallel run
    threads_per_worker : int, optional
        The number of threads per worker for parallel run
    progress_bar : bool
        Flag for showing progress bar
    cluster_args : dict, optional
        Explicit arguments for the cluster setup
    client_args : dict, optional
        Explicit arguments for the client setup
    verbosity : int
        The verbosity level, 0 = silent

    Attributes
    ----------
    scheduler : str, optional
        The dask scheduler choice
    progress_bar : bool
        Flag for showing progress bar
    cluster_args : dict, optional
        Explicit arguments for the cluster setup
    client_args : dict, optional
        Explicit arguments for the client setup
    verbosity : int
        The verbosity level, 0 = silent

    """

    def __init__(
            self, 
            scheduler=None, 
            n_workers=None, 
            threads_per_worker=None,
            processes=True,
            cluster_args=None,
            client_args={},
            progress_bar=False,
            verbosity=1,
        ):

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
        elif (
            n_workers is not None or threads_per_worker is not None or processes is not None
        ):
            raise KeyError("Cannot handle 'n_workers', 'threads_per_worker' or 'processes' arguments if 'cluster_args' are provided")
        else:
            self.cluster_args = cluster_args
        
        if scheduler is None and (
            n_workers is not None or threads_per_worker is not None or processes is not None or cluster_args is not None
        ):
            self.scheduler = "distributed"

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
        func : Function
            The function to be run
        args : tuple
            The function arguments
        kwargs : dict
            The function keyword arguments

        Returns
        -------
        results : Any
            The functions return value

        """
        # parallel run:
        if self.scheduler == "distributed":

            self.print("Launching dask cluster..")

            with LocalCluster(**self.cluster_args) as cluster:
                with Client(cluster, **self.client_args) as client:

                    self.print(cluster)
                    self.print(f"Dashboard: {client.dashboard_link}\n")

                    results = func(*args, **kwargs)

                    self.print("\n\nShutting down dask cluster")

        # serial run:
        else:
            with dask.config.set(scheduler=self.scheduler):
                
                if self.progress_bar:
                    with ProgressBar():
                        results = func(*args, **kwargs)
                else:
                    results = func(*args, **kwargs)
        
        return results
        