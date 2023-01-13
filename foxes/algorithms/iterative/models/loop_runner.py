from copy import deepcopy

from foxes.core import FarmDataModelList


class LoopRunner(FarmDataModelList):
    """
    This model runs the loop of iterations
    within each individual chunk.

    Parameters
    ----------
    mlist : foxes.core.FarmDataModelList
        The models to be iterated
    conv : foxes.algorithm.iterative.models.ConvCrit
        The convergence criteria
    model_wflag : list of bool, optional
        True for models that should be run during wake iteration
    verbosity : int
        The verbosity level, 0 = silent

    Attributes
    ----------
    conv : foxes.algorithm.iterative.models.ConvCrit
        The convergence criteria
    models : list of foxes.core.FarmDataModel
        The model list
    model_wflag : list of bool
        True for models that should be run during wake iterations
    verbosity : int
        The verbosity level, 0 = silent

    """
    def __init__(self, conv, models=[], model_wflag=None, verbosity=0):
        super().__init__(models=models)
        self.conv = conv
        self.verbosity=verbosity
        self.model_wflag = [False for m in models] if model_wflag is None else model_wflag

    def append(self, model, wflag=False):
        """
        Add a model to the list

        Parameters
        ----------
        model : foxes.core.FarmDataModel
            The model to add
        wflag : bool
            True if model should be run during wake iterations

        """
        super().append(model)
        self.model_wflag.append(wflag)

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
        fdata0 = None
        it = 0
        while True:

            if self.verbosity > 0:
                print(f"\n{self.name}: Running iteration {it}\n")

            # run all models at first iteration:
            if fdata0 is None:
                results = super().calculate(algo, mdata, fdata, parameters)
                fdata.update(results)
            
            # only run wake relevant models after first iteration:
            else:
                for mi, m in enumerate(self.models):
                    if self.model_wflag[mi]:
                        results = m.calculate(algo, mdata, fdata, **parameters[mi])
                        fdata.update(results)
            del results
            
            if self.conv.check_converged(self, fdata0, fdata, verbosity=self.verbosity):
                break
            else:
                fdata0 = deepcopy(fdata)
                it += 1
        
        return {v: fdata[v] for v in self.output_farm_vars(algo)}
