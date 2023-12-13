from foxes.core import FarmDataModel


class URelax(FarmDataModel):
    """
    Apply under-relaxation to selected variables
    during iterations.

    Attributes
    ----------
    urel: dict
        The variables and their under-relaxation
        factors bewteen 0 and 1

    :group: algorithms.iterative.models

    """

    def __init__(self, **urel):
        """
        Constructor.

        Parameters
        ----------
        urel: dict
            The variables and their under-relaxation
            factors bewteen 0 and 1

        """
        super().__init__()
        self.urel = urel
        self.name += "_" + "_".join(list(urel.keys()))

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
        return list(self.urel.keys())

    def calculate(self, algo, mdata, fdata):
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

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        i0 = fdata.states_i0(counter=True, algo=algo)
        i1 = i0 + fdata.n_states
        pres = algo.prev_farm_results

        out = {}
        for v, u in self.urel.items():
            if u > 0 and pres is not None:
                odata = pres[v].to_numpy()[i0:i1]
                out[v] = u * odata + (1 - u) * fdata[v]
            else:
                out[v] = fdata[v]

        return out
