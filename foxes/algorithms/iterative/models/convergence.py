from abc import ABCMeta, abstractmethod
import numpy as np

from foxes.utils import delta_wd
import foxes.variables as FV
import foxes.constants as FC

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

        self._deltas = None
        self._conv_states = None
        self.__no_subs = False

    def disable_subsets(self, no_subs=True):
        """
        Disable subset state selection in iterative algorithm.

        This is needed if the convergence criterion requires
        all states to be calculated in each iteration.

        Parameters
        ----------
        no_subs: bool
            Disable subsets flag

        """
        self.__no_subs = no_subs

    @property
    def no_subs(self):
        """
        Get the disable subsets flag.

        Returns
        -------
        no_subs: bool
            Disable subsets flag

        """
        return self.__no_subs

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

    @property
    def deltas(self):
        """
        Get the most recent evaluation deltas.

        Returns
        -------
        deltas: dict
            The most recent evaluation deltas

        """
        return self._deltas
    
    @property
    def conv_states(self):
        """
        Get the convergence state per state.

        Returns
        -------
        conv_states: numpy.ndarray, bool
            The convergence state per state

        """
        return None if self.no_subs else self._conv_states


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
        self._conv_states = None
        for c in self.crits:
            conv = c.check_converged(algo, prev_results, results, verbosity)

            if self._conv_states is None:
                self._conv_states = c.conv_states
                self._deltas = c.deltas
            else:
                self._conv_states = self._conv_states & c.conv_states
                self._deltas = {
                    v: max(self._deltas[v], d) for v, d in c.deltas.items()
                }

            if self.failed is None and not conv:
                self._failed = c

        return self._failed is None


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
        n_states = prev_results.sizes[FC.STATE]
        self._deltas = {}
        self._conv_states = np.ones(n_states, dtype=bool)
        for v, lim in self.limits.items():
            x0 = prev_results[v].to_numpy()
            x = results[v].to_numpy()
            if v in self.wd_vars:
                a = np.abs(delta_wd(x0, x))
            else:
                a = np.abs(x - x0)
            self._deltas[v] = np.max(a)
            check = self._deltas[v]
            self._conv_states = self._conv_states & np.all(a <= lim, axis=1)
            ok = ok and (check <= lim)

            if verbosity > 0:
                r = "FAILED" if check > lim else "OK"
                print(f"  {v:<{L}}: delta = {check:.3e}, lim = {lim:.3e}  --  {r}")
        
        if verbosity > 0:
            print(f"Converged states: {self._conv_states.sum()}/{n_states}")

        if ok:
            self._conv_states = None
            
        return ok


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
