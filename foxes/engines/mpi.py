from foxes.utils import import_module

from .futures import ProcessEngine


class MPIEngine(ProcessEngine):
    """
    The MPI engine for foxes calculations.

    Examples
    --------
    Run command, e.g. for 12 processors and a script run.py:

    >>> mpiexec -n 12 -m mpi4py.futures run.py

    :group: engines

    """


    def _create_pool(self):
        """Creates the pool"""
        mpi4py_futures = import_module(
            "mpi4py.futures",
            pip_hint="pip install mpi4py",
            conda_hint="conda install mpi4py -c conda-forge",
        )
        MPIPoolExecutor = mpi4py_futures.MPIPoolExecutor

        pargs = dict(use_pkl5=True)
        pargs.update(self.pool_args)
        self._pool = MPIPoolExecutor(max_workers=self.n_workers, **pargs)
