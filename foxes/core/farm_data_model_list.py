from .farm_data_model import FarmDataModel


class FarmDataModelList(FarmDataModel):
    """
    A list of farm data models.

    By using the FarmDataModelList the models'
    `calculate` functions are called together
    under one common call of xarray's `apply_ufunc`.

    Parameters
    ----------
    models : list of foxes.core.FarmDataModel
        The model list

    Attributes
    ----------
    models : list of foxes.core.FarmDataModel
        The model list

    """

    def __init__(self, models=[]):
        super().__init__()
        self.models = models

    def output_farm_vars(self, algo):
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
        ovars = []
        for m in self.models:
            ovars += m.output_farm_vars(algo)
        return list(dict.fromkeys(ovars))

    def model_input_data(self, algo):
        """
        The model input data, as needed for the
        calculation.

        This function should specify all data
        that depend on the loop variable (e.g. state),
        or that are intended to be shared between chunks.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        idata : dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        idata = super().model_input_data(algo)
        for m in self.models:
            hidata = m.model_input_data(algo)
            idata["coords"].update(hidata["coords"])
            idata["data_vars"].update(hidata["data_vars"])

        return idata

    def initialize(self, algo, parameters=None, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        parameters : list of dict, optional
            A list of parameter dicts, one for each model
        verbosity : int
            The verbosity level, 0 means silent

        """
        if parameters is None:
            parameters = [{}] * len(self.models)
        elif not isinstance(parameters, list):
            raise ValueError(
                f"{self.name}: Wrong parameters type, expecting list, got {type(parameters).__name__}"
            )
        elif len(parameters) != len(self.models):
            raise ValueError(
                f"{self.name}: Wrong parameters length, expecting list with {len(self.models)} entries, got {len(parameters)}"
            )

        for mi, m in enumerate(self.models):
            if not m.initialized:
                if verbosity > 0:
                    print(f"{self.name}, sub-model '{m.name}': Initializing")
                m.initialize(algo, verbosity=verbosity, **parameters[mi])

        super().initialize(algo)

    def calculate(self, algo, mdata, fdata, parameters=[]):
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
        parameters : list of dict, optional
            A list of parameter dicts, one for each model

        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        if parameters is None:
            parameters = [{}] * len(self.models)
        elif not isinstance(parameters, list):
            raise ValueError(
                f"{self.name}: Wrong parameters type, expecting list, got {type(parameters).__name__}"
            )
        elif len(parameters) != len(self.models):
            raise ValueError(
                f"{self.name}: Wrong parameters length, expecting list with {len(self.models)} entries, got {len(parameters)}"
            )

        for mi, m in enumerate(self.models):
            # print("MLIST VARS BEFORE",m.name,list(fdata.keys()),parameters[mi])
            res = m.calculate(algo, mdata, fdata, **parameters[mi])
            fdata.update(res)

        return {v: fdata[v] for v in self.output_farm_vars(algo)}

    def finalize(self, algo, results, parameters=[], verbosity=0, clear_mem=False):
        """
        Finalizes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        parameters : list of dict, optional
            A list of parameter dicts, one for each model
        verbosity : int
            The verbosity level, 0 means silent
        clear_mem : bool
            Flag for deleting model data and
            resetting initialization flag

        """
        if parameters is None:
            parameters = [{}] * len(self.models)
        elif not isinstance(parameters, list):
            raise ValueError(
                f"{self.name}: Wrong parameters type, expecting list, got {type(parameters).__name__}"
            )
        elif len(parameters) != len(self.models):
            raise ValueError(
                f"{self.name}: Wrong parameters length, expecting list with {len(self.models)} entries, got {len(parameters)}"
            )

        for mi, m in enumerate(self.models):
            if verbosity > 0:
                print(f"{self.name}, sub-model '{m.name}': Finalizing")
            m.finalize(algo, results, **parameters[mi])

        if clear_mem:
            self.models = None

        super().finalize(algo, results, clear_mem)
