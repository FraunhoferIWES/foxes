from foxes.core import TurbineType

class CalculatorType(TurbineType):
    """
    Direct data infusion by a user function.

    :group: models.turbine_types

    """

    def __init__(
        self,
        func,
        out_vars,
        *args,
        needs_rews2=False,
        needs_rews3=False,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        func: callable
            The function to calculate farm variables, should have the signature:
            f(algo, mdata, fdata, st_sel) -> dict, where the keys are
            output variable names and the values are numpy.ndarrays
            with shape (n_states, n_turbines).

            Beware that the turbine ordering in fdata is in downwind order,
            hence external data X of shape (n_states, n_turbines) in farm order
            needs to be reordered by X[ssel, order] with
            ssel = fdata[FV.ORDER_SSEL], order = fdata[FV.ORDER]
            before using it in combination with fdata variables.
        out_vars: list of str
            The output variables of the function
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
        self._func = func
        self._ovars = out_vars
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
        return self._ovars

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
        self.ensure_output_vars(algo, fdata)
        return self._func(algo, mdata, fdata, st_sel)
