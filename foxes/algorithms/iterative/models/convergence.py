import numpy as np
from abc import ABCMeta, abstractmethod

import foxes.variables as FV
from foxes.utils import delta_wd

class ConvCrit(metaclass=ABCMeta):
    """
    Abstract base class for convergence criteria

    Parameters
    ----------
    name : str, optional
        The convergence criteria name
    
    Attribute
    ---------
    name : str, optional
        The convergence criteria name

    """
    def __init__(self, name=None):
        self.name = name if name is not None else type(self).__name__
        
    @abstractmethod
    def check_converged(self, algo, fdata0, fdata1, verbosity=0):
        """
        Check convergence criteria.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        fdata0 : foxes.core.Data
            The farm data results of previous 
            iteration, or None if first
        fdata1 : foxes.core.Data
            The farm data results of current 
            iteration, or None if first
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        convergence : bool
            Convergence flag, true if converged

        """
        pass

class ConvCritList(ConvCrit):
    """
    A list of convergence criteria

    Parameters
    ----------
    crits : list of ConvCrit
        The criteria
    name : str, optional
        The convergence criteria name

    Attributes
    ----------
    crits : list of ConvCrit
        The criteria

    """
    def __init__(self, crits=[], name=None):
        super().__init__(name)
        self.crits = crits
    
    def add_crit(self, crit):
        """
        Add a convergence criterion

        Parameters
        ----------
        crit : ConvCrit
            The criterion

        """
        self.crits.append(crit)
    
    def check_converged(self, algo, fdata0, fdata1, verbosity=0):
        """
        Check convergence criteria.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        fdata0 : foxes.core.Data
            The farm data results of previous 
            iteration, or None if first
        fdata1 : foxes.core.Data
            The farm data results of current 
            iteration, or None if first
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        convergence : bool
            Convergence flag, true if converged

        """
        for c in self.crits:
            if not c.check_converged(algo, fdata0, fdata1, verbosity):
                return False
        
        return True
    
class ConvVarDelta(ConvCrit):
    """
    Requires convergence of a selection of variables.

    Parameters
    ----------
    limits : dict
        The convergence limits. Keys: variables str,
        values: float values
    wd_vars : list of str, optional
        The wind direction type variables (unit deg)
    name : str, optional
        The convergence criteria name

    Attributes
    ----------
    limits : dict
        The convergence limits. Keys: variables str,
        values: float values
    wd_vars : list of str
        The wind direction type variables (unit deg)

    """
    def __init__(self, limits, wd_vars=None, name=None):
        super().__init__(name)
        self.limits = limits
        if wd_vars is None:
            self.wd_vars = [FV.WD, FV.AMB_WD, FV.YAW, FV.AMB_YAW]
        else:
            self.wd_vars = wd_vars

    def check_converged(self, algo, fdata0, fdata1, verbosity=0):
        """
        Check convergence criteria.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        fdata0 : foxes.core.Data
            The farm data results of previous 
            iteration, or None if first
        fdata1 : foxes.core.Data
            The farm data results of current 
            iteration, or None if first
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        convergence : bool
            Convergence flag, true if converged

        """
        if fdata0 is None:
            return False
        
        if verbosity > 0:
            print(f"\n{self.name}: Convergence check")
        
        ok = True
        for v, lim in self.limits.items():

            if v in self.wd_vars:
                check = np.max(np.abs(delta_wd(fdata0[v], fdata1[v])))
            else:
                check = np.max(np.abs(fdata1[v] - fdata0[v]))
            ok = ok and (check <= lim)
            
            if verbosity > 0:
                r = "FAILED" if check > lim else "OK"
                print(f"  {v}: delta = {check}, lim = {lim}  --  {r}")
            elif not ok:
                break
            
        return ok

class DefaultConv(ConvVarDelta):
    """
    Default convergence criteria.
    """
    def __init__(self):
        super().__init__({
            FV.REWS: 1e-5,
            FV.WD: 1e-4,
            FV.TI: 1e-6,
        })
