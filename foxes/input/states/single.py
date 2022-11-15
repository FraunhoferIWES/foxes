import numpy as np

from foxes.core import States, VerticalProfile
import foxes.variables as FV
import foxes.constants as FC


class SingleStateStates(States):
    """
    A single uniform state.

    Parameters
    ----------
    ws : float
        The wind speed
    wd : float
        The wind direction
    ti : float, optional
        The TI value
    rho : float, optional
        The air density
    profiles : dict, optional
        Key: output variable name str, Value: str or dict
        or `foxes.core.VerticalProfile`
    profdata : dict, optional
        Additional data for profiles

    Attributes
    ----------
    ws : float
        The wind speed
    wd : float
        The wind direction
    ti : float
        The TI value
    rho : float
        The air density
    profdicts : dict
        Key: output variable name str, Value: str or dict
        or `foxes.core.VerticalProfile`
    profdata : dict,
        Additional data for profiles

    """

    def __init__(self, ws, wd, ti=None, rho=None, profiles={}, **profdata):
        super().__init__()
        self.ws = ws
        self.wd = wd
        self.ti = ti
        self.rho = rho
        self.profdicts = profiles
        self.profdata = profdata

    def initialize(self, algo, states_sel=None, states_loc=None, verbosity=1):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        states_sel : slice or range or list of int, optional
            States subset selection
        states_loc : list, optional
            State index selection via pandas loc function
        verbosity : int
            The verbosity level

        """
        super().initialize(algo, verbosity=verbosity)

        self._profiles = {}
        for v, d in self.profdicts.items():
            if isinstance(d, str):
                self._profiles[v] = VerticalProfile.new(d)
            elif isinstance(d, VerticalProfile):
                self._profiles[v] = d
            elif isinstance(d, dict):
                t = d.pop("type")
                self._profiles[v] = VerticalProfile.new(t, **d)
            else:
                raise TypeError(
                    f"States '{self.name}': Wrong profile type '{type(d).__name__}' for variable '{v}'. Expecting VerticalProfile, str or dict"
                )

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return 1

    def output_point_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars : list of str
            The output variable names

        """
        out = []
        if self.ws is not None:
            out.append(FV.WS)
        if self.wd is not None:
            out.append(FV.WD)
        if self.ti is not None:
            out.append(FV.TI)
        if self.rho is not None:
            out.append(FV.RHO)
        return out

    def weights(self, algo):
        """
        The statistical weights of all states.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        weights : numpy.ndarray
            The weights, shape: (n_states, n_turbines)

        """
        return np.ones((1, algo.n_turbines), dtype=FC.DTYPE)

    def calculate(self, algo, mdata, fdata, pdata):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        pdata : foxes.core.Data
            The point data

        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """

        if self.ws is not None:
            pdata[FV.WS] = np.full(
                (pdata.n_states, pdata.n_points), self.ws, dtype=FC.DTYPE
            )
        if self.wd is not None:
            pdata[FV.WD] = np.full(
                (pdata.n_states, pdata.n_points), self.wd, dtype=FC.DTYPE
            )
        if self.ti is not None:
            pdata[FV.TI] = np.full(
                (pdata.n_states, pdata.n_points), self.ti, dtype=FC.DTYPE
            )
        if self.rho is not None:
            pdata[FV.RHO] = np.full(
                (pdata.n_states, pdata.n_points), self.rho, dtype=FC.DTYPE
            )

        z = pdata[FV.POINTS][:, :, 2]
        if len(self._profiles):
            z = pdata[FV.POINTS][:, :, 2]
            for k, v in self.profdata.items():
                pdata[k] = v
            for v, p in self._profiles.items():
                pres = p.calculate(pdata, z)
                pdata[v] = pres

        return {v: pdata[v] for v in self.output_point_vars(algo)}
