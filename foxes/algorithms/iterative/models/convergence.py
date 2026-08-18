from __future__ import annotations

from abc import ABCMeta, abstractmethod
import numpy as np
from typing import TYPE_CHECKING

from foxes.utils import delta_wd
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from xarray import Dataset
    from foxes.core.algorithm import Algorithm


class ConvCrit(metaclass=ABCMeta):
    """
    Abstract base class for convergence criteria

    Attributes
    ----------
    name
        The convergence criteria name


    """

    def __init__(self, name: str | None = None) -> None:
        """
        Constructor.

        Parameters
        ----------
        name
            The convergence criteria name

        """
        self.name = name if name is not None else type(self).__name__

        self._deltas: dict[str, float] | None = None
        self._conv_states: np.ndarray | None = None
        self.__no_subs = False

    def disable_subsets(self, no_subs: bool = True) -> None:
        """
        Disable subset state selection in iterative algorithm.

        This is needed if the convergence criterion requires
        all states to be calculated in each iteration.

        Parameters
        ----------
        no_subs
            Disable subsets flag

        """
        self.__no_subs = no_subs

    @property
    def no_subs(self) -> bool:
        """
        Get the disable subsets flag.

        Returns
        -------
        no_subs
            Disable subsets flag

        """
        return self.__no_subs

    @abstractmethod
    def check_converged(
        self,
        algo: Algorithm,
        prev_results: Dataset | None,
        results: Dataset,
        verbosity: int = 0,
    ) -> bool:
        """
        Check convergence criteria.

        Parameters
        ----------
        algo
            The calculation algorithm
        prev_results
            The farm results of previous
            iteration, or None if first
        results
            The farm results of current
            iteration
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        convergence
            Convergence flag, true if converged

        """
        pass

    @property
    def deltas(self) -> dict[str, float] | None:
        """
        Get the most recent evaluation deltas.

        Returns
        -------
        deltas
            The most recent evaluation deltas

        """
        return self._deltas

    @property
    def conv_states(self) -> np.ndarray | None:
        """
        Get the convergence state per state.

        Returns
        -------
        conv_states
            The convergence state per state

        """
        return None if self.no_subs else self._conv_states


class ConvCritList(ConvCrit):
    """
    Combines multiple convergence criteria.

    Attributes
    ----------
    crits
        The criteria


    """

    def __init__(self, crits: list[ConvCrit] = [], name: str | None = None) -> None:
        """
        Constructor.

        Parameters
        ----------
        crits
            The criteria
        name
            The convergence criteria name

        """
        super().__init__(name)
        self.crits = crits
        self._failed: ConvCrit | None = None

    @property
    def failed(self) -> ConvCrit | None:
        return self._failed

    def add_crit(self, crit: ConvCrit) -> None:
        """
        Add a convergence criterion

        Parameters
        ----------
        crit
            The criterion

        """
        self.crits.append(crit)

    def check_converged(
        self,
        algo: Algorithm,
        prev_results: Dataset | None,
        results: Dataset,
        verbosity: int = 0,
    ) -> bool:
        """
        Check convergence criteria.

        Parameters
        ----------
        algo
            The calculation algorithm
        prev_results
            The farm results of previous
            iteration, or None if first
        results
            The farm results of current
            iteration
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        convergence
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
                assert self._conv_states is not None
                assert c.conv_states is not None
                assert self._deltas is not None
                assert c.deltas is not None
                self._conv_states = self._conv_states & c.conv_states
                self._deltas = {v: max(self._deltas[v], d) for v, d in c.deltas.items()}

            if self.failed is None and not conv:
                self._failed = c

        return self._failed is None


class ConvVarDelta(ConvCrit):
    """
    Requires convergence of a selection of variables.

    Attributes
    ----------
    limits
        The convergence limits. Keys: variables str,
        values are convergence thresholds
    wd_vars
        The wind direction type variables (unit deg)


    """

    def __init__(
        self,
        limits: dict[str, float],
        wd_vars: list[str] | None = None,
        name: str | None = None,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        limits
            The convergence limits. Keys: variables str,
            values are convergence thresholds
        wd_vars
            The wind direction type variables (unit deg)
        name
            The convergence criteria name

        """
        super().__init__(name)
        self.limits = limits
        if wd_vars is None:
            self.wd_vars = [FV.WD, FV.AMB_WD, FV.YAW, FV.AMB_YAW]
        else:
            self.wd_vars = wd_vars

    def check_converged(
        self,
        algo: Algorithm,
        prev_results: Dataset | None,
        results: Dataset,
        verbosity: int = 0,
    ) -> bool:
        """
        Check convergence criteria.

        Parameters
        ----------
        algo
            The calculation algorithm
        prev_results
            The farm results of previous
            iteration, or None if first
        results
            The farm results of current
            iteration
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        convergence
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


    """

    def __init__(self) -> None:
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
