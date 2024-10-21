from foxes.utils import import_module

from .pool import PoolEngine
from .futures import ProcessEngine


def load_mpi():
    """On-demand loading of the mpi4py package"""
    global MPIPoolExecutor
    MPIPoolExecutor = import_module(
        "mpi4py.futures", hint="pip install mpi4py"
    ).MPIPoolExecutor


class MPIEngine(ProcessEngine):
    """
    The MPI engine for foxes calculations.

    Examples
    --------
    Run command, e.g. for 12 processors and a script run.py:

    >>> mpiexec -n 12 -m mpi4py.futures run.py

    :group: engines

    """

    def initialize(self):
        """
        Initializes the engine.
        """
        load_mpi()
        PoolEngine.initialize(self)

    def _create_pool(self):
        """Creates the pool"""
        self._pool = MPIPoolExecutor(max_workers=self.n_procs)
