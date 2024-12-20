from abc import ABCMeta, abstractmethod
import numpy as np

import foxes.variables as FV
from foxes.utils import delta_wd


class ConvCrit(metaclass=ABCMeta):
    """
    Abstract base class for convergence criteria

    Attributes
    ----------
    name: str, optional
        The convergence criteria name

    :group: algorithms.iterative.models

    """

    def __init__(self, name=None):
        """
        Constructor.

        Parameters
        ----------
        name: str, optional
            The convergence criteria name

        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def check_converged(self, algo, prev_results, results, verbosity=0):
        """
        Check convergence criteria.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        prev_results: xarray.Dataset
            The farm results of previous
            iteration, or None if first
        results: xarray.Dataset
            The farm results of current
            iteration
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        convergence: bool
            Convergence flag, true if converged

        """
        pass

    @abstractmethod
    def get_deltas(self):
        """
        Get the most recent evaluation deltas.

        Returns
        -------
        deltas: dict
            The most recent evaluation deltas

        """
        pass


class ConvCritList(ConvCrit):
    """
    A list of convergence criteria

    Attributes
    ----------
    crits: list of ConvCrit
        The criteria

    :group: algorithms.iterative.models

    """

    def __init__(self, crits=[], name=None):
        """
        Constructor.

        Parameters
        ----------
        crits: list of ConvCrit
            The criteria
        name: str, optional
            The convergence criteria name

        """
        super().__init__(name)
        self.crits = crits

    def add_crit(self, crit):
        """
        Add a convergence criterion

        Parameters
        ----------
        crit: ConvCrit
            The criterion

        """
        self.crits.append(crit)

    def check_converged(self, algo, prev_results, results, verbosity=0):
        """
        Check convergence criteria.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        prev_results: xarray.Dataset
            The farm results of previous
            iteration, or None if first
        results: xarray.Dataset
            The farm results of current
            iteration
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        convergence: bool
            Convergence flag, true if converged

        """
        self._failed = None
        for c in self.crits:
            if not c.check_converged(algo, prev_results, results, verbosity):
                self._failed = c
                return False

        return True

    def get_deltas(self):
        """
        Get the most recent evaluation deltas.

        Returns
        -------
        deltas: dict
            The most recent evaluation deltas

        """
        if self._failed is not None:
            return self._failed.get_deltas()
        return {}


class ConvVarDelta(ConvCrit):
    """
    Requires convergence of a selection of variables.

    Attributes
    ----------
    limits: dict
        The convergence limits. Keys: variables str,
        values: float values
    wd_vars: list of str
        The wind direction type variables (unit deg)

    :group: algorithms.iterative.models

    """

    def __init__(self, limits, wd_vars=None, name=None):
        """
        Constructor.

        Parameters
        ----------
        limits: dict
            The convergence limits. Keys: variables str,
            values: float values
        wd_vars: list of str, optional
            The wind direction type variables (unit deg)
        name: str, optional
            The convergence criteria name

        """
        super().__init__(name)
        self.limits = limits
        if wd_vars is None:
            self.wd_vars = [FV.WD, FV.AMB_WD, FV.YAW, FV.AMB_YAW]
        else:
            self.wd_vars = wd_vars

        self._deltas = {}

    def check_converged(self, algo, prev_results, results, verbosity=0):
        """
        Check convergence criteria.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        prev_results: xarray.Dataset
            The farm results of previous
            iteration, or None if first
        results: xarray.Dataset
            The farm results of current
            iteration
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        convergence: bool
            Convergence flag, true if converged

        """
        if prev_results is None:
            return False

        if verbosity > 0:
            print(f"\n{self.name}: Convergence check")
            L = max([len(v) for v in self.limits.keys()])

        ok = True
        self._deltas = {}
        for v, lim in self.limits.items():
            x0 = prev_results[v].to_numpy()
            x = results[v].to_numpy()
            if v in self.wd_vars:
                self._deltas[v] = np.max(np.abs(delta_wd(x0, x)))
            else:
                self._deltas[v] = np.max(np.abs(x - x0))
            check = self._deltas[v]
            ok = ok and (check <= lim)

            if verbosity > 0:
                r = "FAILED" if check > lim else "OK"
                print(f"  {v:<{L}}: delta = {check:.3e}, lim = {lim:.3e}  --  {r}")
            elif not ok:
                break

        return ok

    def get_deltas(self):
        """
        Get the most recent evaluation deltas.

        Returns
        -------
        deltas: dict
            The most recent evaluation deltas

        """
        return self._deltas


class DefaultConv(ConvVarDelta):
    """
    Default convergence criteria.

    :group: algorithms.iterative.models

    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__(
            {
                FV.REWS: 1e-6,
                FV.TI: 1e-7,
                FV.CT: 1e-7,
            }
        )
