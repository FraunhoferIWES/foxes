from foxes.core import TurbineType


class NullType(TurbineType):
    """
    A turbine type that does not compute any data.

    :group: models.turbine_types

    """

    def __init__(
        self,
        *args,
        needs_rews2=False,
        needs_rews3=False,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Additional parameters for TurbineType class
        needs_rews2: bool
            Flag for runs that require the REWS2 variable
        needs_rews3: bool
            Flag for runs that require the REWS3 variable
        kwargs: dict, optional
            Additional parameters for TurbineType class

        """
        super().__init__(*args, **kwargs)
        self._rews2 = needs_rews2
        self._rews3 = needs_rews3

    def needs_rews2(self):
        """
        Returns flag for requiring REWS2 variable

        Returns
        -------
        flag: bool
            True if REWS2 is required

        """
        return self._rews2

    def needs_rews3(self):
        """
        Returns flag for requiring REWS3 variable

        Returns
        -------
        flag: bool
            True if REWS3 is required

        """
        return self._rews3

    def output_farm_vars(self, algo):
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
        return []

    def calculate(self, algo, mdata, fdata, st_sel):
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        st_sel: numpy.ndarray of bool
            The state-turbine selection,
            shape: (n_states, n_turbines)

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        return {}
