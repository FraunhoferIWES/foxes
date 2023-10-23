import numpy as np

from foxes.core import States, VerticalProfile
import foxes.variables as FV
import foxes.constants as FC


class SingleStateStates(States):
    """
    A single uniform state.

    Attributes
    ----------
    ws: float
        The wind speed
    wd: float
        The wind direction
    ti: float
        The TI value
    rho: float
        The air density
    profdicts: dict
        Key: output variable name str, Value: str or dict
        or `foxes.core.VerticalProfile`
    profdata: dict,
        Additional data for profiles

    :group: input.states

    """

    def __init__(self, ws=None, wd=None, ti=None, rho=None, profiles={}, **profdata):
        """
        Constructor.

        Parameters
        ----------
        ws: float, optional
            The wind speed
        wd: float, optional
            The wind direction
        ti: float, optional
            The TI value
        rho: float, optional
            The air density
        profiles: dict, optional
            Key: output variable name str, Value: str or dict
            or `foxes.core.VerticalProfile`
        profdata: dict, optional
            Additional data for profiles

        """
        super().__init__()
        self.ws = ws
        self.wd = wd
        self.ti = ti
        self.rho = rho
        self.profdicts = profiles
        self.profdata = profdata

        if (
            ws is None
            and wd is None
            and ti is None
            and rho is None
            and not len(profiles)
        ):
            raise KeyError(
                f"Expecting at least one parameter: ws, wd, ti, rho, profiles"
            )

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return list(self._profiles.values())

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
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
        super().initialize(algo, verbosity)

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
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars: list of str
            The output variable names

        """
        out = set()
        if self.ws is not None:
            out.add(FV.WS)
        if self.wd is not None:
            out.add(FV.WD)
        if self.ti is not None:
            out.add(FV.TI)
        if self.rho is not None:
            out.add(FV.RHO)
        out.update(list(self._profiles.keys()))

        return list(out)

    def weights(self, algo):
        """
        The statistical weights of all states.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        weights: numpy.ndarray
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
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The point data

        Returns
        -------
        results: dict
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

        if len(self._profiles):
            z = pdata[FC.POINTS][:, :, 2]
            for k, v in self.profdata.items():
                pdata[k] = v
            for v, p in self._profiles.items():
                pres = p.calculate(pdata, z)
                pdata[v] = pres

        return {v: pdata[v] for v in self.output_point_vars(algo)}
