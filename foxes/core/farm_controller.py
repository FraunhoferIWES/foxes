import numpy as np

from .farm_data_model_list import FarmDataModelList
from .farm_data_model import FarmDataModel
from .turbine_model import TurbineModel
from .turbine_type import TurbineType
import foxes.constants as FC
import foxes.variables as FV


class FarmController(FarmDataModel):
    """
    Analyses selected turbine models and handles their call.

    Parameters
    ----------
    pars : dict
        Parameters for the turbine models, stored
        under their respective name

    Attributes
    ----------
    turbine_types : list of foxes.core.TurbineType
        The turbine type of each turbine
    turbine_model_names : list of str
        Names of all turbine models found in the farm
    turbine_model_sels : numpy.ndarray of bool
        Selection flags for all turbine models,
        shape: (n_states, n_turbines, n_models)
    pre_rotor_models : foxes.core.FarmDataModelList
        The turbine models with pre-rotor flag
    post_rotor_models : foxes.core.FarmDataModelList
        The turbine models without pre-rotor flag
    pars : dict
        Parameters for the turbine models, stored
        under their respecitve name

    """

    def __init__(self, pars={}):
        super().__init__()

        self.turbine_types = None
        self.turbine_model_names = None
        self.turbine_model_sels = None
        self.pre_rotor_models = None
        self.post_rotor_models = None

        self.pars = pars

    def set_pars(self, model_name, init_pars, calc_pars, final_pars):
        """
        Set parameters for a turbine model

        Parameters
        ----------
        model_name : str
            Name of the model
        init_pars : dict
            Parameters for initialization
        calc_pars : dict
            Parameters for calculation
        final_pars : dict
            Parameters for finalization

        """
        self.pars[model_name] = {
            "init": init_pars,
            "calc": calc_pars,
            "final": final_pars,
        }

    def _analyze_models(self, algo, pre_rotor, models):
        """
        Helper function for model analysis
        """
        tmodels = []
        tmsels = []
        mnames = [[m.name for m in mlist] for mlist in models]
        tmis = np.zeros(algo.n_turbines, dtype=FC.ITYPE)
        news = True
        while news:
            news = False

            for ti, mlist in enumerate(models):

                if tmis[ti] < len(mlist):

                    mname = mnames[ti][tmis[ti]]
                    isnext = True
                    for tj, jnames in enumerate(mnames):
                        if (
                            tj != ti
                            and mname in jnames
                            and tmis[tj] < len(jnames)
                            and jnames[tmis[tj]] != mname
                        ):
                            isnext = False
                            break

                    if isnext:

                        m = models[ti][tmis[ti]]
                        tmodels.append(m)

                        tsel = np.zeros((algo.n_states, algo.n_turbines), dtype=bool)
                        for tj, jnames in enumerate(mnames):
                            mi = tmis[tj]
                            if mi < len(jnames) and jnames[mi] == mname:
                                ssel = algo.farm.turbines[tj].mstates_sel[mi]
                                tsel[:, tj] = True if ssel is None else ssel
                                tmis[tj] += 1
                        tmsels.append(tsel)

                        news = True
                        break

        if pre_rotor:
            self.pre_rotor_models = FarmDataModelList(tmodels)
            self.pre_rotor_models.name = f"{self.name}_prer"
            mtype = "pre-rotor"
        else:
            self.post_rotor_models = FarmDataModelList(tmodels)
            self.post_rotor_models.name = f"{self.name}_postr"
            mtype = "post-rotor"

        for ti, t in enumerate(algo.farm.turbines):
            if tmis[ti] != len(models[ti]):
                raise ValueError(
                    f"Turbine {ti}, {t.name}: Could not find turbine model order that includes all {mtype} turbine models, missing {t.models[tmis[ti]:]}"
                )

        return [m.name for m in tmodels], tmsels

    def collect_models(self, algo):
        """
        Analyze and gather turbine models, based on the
        turbines of the wind farm.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        """

        # check turbine models, and find turbine types and pre/post-rotor models:
        self.turbine_types = [None for t in algo.farm.turbines]
        prer_models = [[] for t in algo.farm.turbines]
        postr_models = [[] for t in algo.farm.turbines]
        for ti, t in enumerate(algo.farm.turbines):
            prer = None
            for mi, mname in enumerate(t.models):

                istype = False
                if mname in algo.mbook.turbine_types:
                    m = algo.mbook.turbine_types[mname]
                    if not isinstance(m, TurbineType):
                        raise TypeError(
                            f"Model {mname} type {type(m).__name__} is not derived from {TurbineType.__name__}"
                        )
                    istype = True
                elif mname in algo.mbook.turbine_models:
                    m = algo.mbook.turbine_models[mname]
                    if not isinstance(m, TurbineModel):
                        raise TypeError(
                            f"Model {mname} type {type(m).__name__} is not derived from {TurbineModel.__name__}"
                        )
                else:
                    raise KeyError(
                        f"Model {mname} not found in model book types or models"
                    )
                if istype:
                    if self.turbine_types[ti] is None:
                        self.turbine_types[ti] = m
                    else:
                        raise ValueError(
                            f"Turbine {ti}, {t.name}: Multiple turbine types found in self.turbine_models list, {self.turbine_types[ti].name} and {mname}"
                        )

                m.name = mname
                if prer is None:
                    prer = m.pre_rotor
                elif not prer and m.pre_rotor:
                    raise ValueError(
                        f"Turbine {ti}, {t.name}: Model is classified as pre-rotor, but following the post-rotor model '{t.models[mi-1]}'"
                    )
                if m.pre_rotor:
                    prer_models[ti].append(m)
                else:
                    postr_models[ti].append(m)

            if self.turbine_types[ti] is None:
                raise ValueError(
                    f"Turbine {ti}, {t.name}: Missing a turbine type model among models {t.models}"
                )

        # analyze models:
        mnames_pre, tmsels_pre = self._analyze_models(
            algo, pre_rotor=True, models=prer_models
        )
        mnames_post, tmsels_post = self._analyze_models(
            algo, pre_rotor=False, models=postr_models
        )
        tmsels = tmsels_pre + tmsels_post
        self.turbine_model_names = mnames_pre + mnames_post
        if len(self.turbine_model_names):
            self.turbine_model_sels = np.stack(tmsels, axis=2)
        else:
            raise ValueError(f"Controller '{self.name}': No turbine model found.")

    def __get_pars(self, algo, models, ptype, mdata=None, st_sel=None, from_data=True):
        """
        Private helper function for gathering model parameters.
        """
        if from_data:
            s = mdata[FV.TMODEL_SELS]
        else:
            if self.turbine_model_names is None:
                self.collect_models(algo)
            s = self.turbine_model_sels
        if st_sel is not None:
            s = s & st_sel[:, :, None]

        pars = [
            {"st_sel": s[:, :, self.turbine_model_names.index(m.name)]} for m in models
        ]
        for mi, m in enumerate(models):
            if m.name in self.pars:
                pars[mi].update(self.pars[m.name][ptype])

        return pars

    def initialize(self, algo, st_sel=None, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        st_sel : numpy.ndarray of bool, optional
            Selection of states and turbines, shape:
            (n_states, n_turbines). None for all.
        verbosity : int
            The verbosity level, 0 means silent

        """
        if self.turbine_model_names is None:
            self.collect_models(algo)

        super().initialize(algo, verbosity=verbosity)

        for s in [self.pre_rotor_models, self.post_rotor_models]:
            pars = self.__get_pars(
                algo, s.models, "init", st_sel=st_sel, from_data=False
            )
            if not s.initialized:
                s.initialize(algo, parameters=pars, verbosity=verbosity)

    def model_input_data(self, algo):
        """
        The model input data, as needed for the
        calculation.

        This function is automatically called during
        initialization. It should specify all data
        that is either state or point dependent, or
        intended to be shared between chunks.

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
        if self.turbine_model_names is None:
            self.collect_models(algo)

        idata = super().model_input_data(algo)
        idata["coords"][FV.TMODELS] = self.turbine_model_names
        idata["data_vars"][FV.TMODEL_SELS] = (
            (FV.STATE, FV.TURBINE, FV.TMODELS),
            self.turbine_model_sels,
        )

        for s in [self.pre_rotor_models, self.post_rotor_models]:
            jdata = s.model_input_data(algo)
            idata["coords"].update(jdata["coords"])
            idata["data_vars"].update(jdata["data_vars"])

        return idata

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
        if self.turbine_model_names is None:
            self.collect_models(algo)
        return list(
            dict.fromkeys(
                self.pre_rotor_models.output_farm_vars(algo)
                + self.post_rotor_models.output_farm_vars(algo)
            )
        )

    def calculate(self, algo, mdata, fdata, pre_rotor, st_sel=None):
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
        pre_rotor : bool
            Flag for running pre-rotor or post-rotor
            models
        st_sel : numpy.ndarray of bool, optional
            Selection of states and turbines, shape:
            (n_states, n_turbines). None for all.

        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        s = self.pre_rotor_models if pre_rotor else self.post_rotor_models
        pars = self.__get_pars(algo, s.models, "calc", mdata, st_sel, from_data=True)
        res = s.calculate(algo, mdata, fdata, parameters=pars)
        self.turbine_model_sels = mdata[FV.TMODEL_SELS]
        return res

    def finalize(self, algo, results, st_sel=None, verbosity=0, clear_mem=False):
        """
        Finalizes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        results : xarray.Dataset
            The calculation results
        st_sel : numpy.ndarray of bool, optional
            Selection of states and turbines, shape:
            (n_states, n_turbines). None for all.
        verbosity : int
            The verbosity level, 0 means silent
        clear_mem : bool
            Flag for deleting model data and
            resetting initialization flag

        """
        for s in [self.pre_rotor_models, self.post_rotor_models]:
            if s is not None and s.initialized:
                pars = self.__get_pars(
                    algo, s.models, "final", st_sel=st_sel, from_data=False
                )
                s.finalize(
                    algo,
                    results=results,
                    parameters=pars,
                    verbosity=verbosity,
                    clear_mem=clear_mem,
                )

        super().finalize(algo, results, clear_mem=clear_mem)
