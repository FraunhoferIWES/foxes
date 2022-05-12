import numpy as np

from foxes.core.farm_data_model_list import FarmDataModelList
from foxes.core.farm_data_model import FarmDataModel
from foxes.core.turbine_model import TurbineModel
from foxes.core.turbine_type import TurbineType
import foxes.constants as FC
import foxes.variables as FV

class FarmController(FarmDataModel):
    
    def __init__(self, pars={}):
        super().__init__()

        self.turbine_types       = None
        self.turbine_model_names = None
        self.turbine_model_sels  = None
        self.pre_rotor_models    = None
        self.post_rotor_models   = None

        self.pars = pars
    
    def set_pars(self, model_name, init_pars, calc_pars, final_pars):
        self.pars[model_name] = {
            "init" : init_pars, 
            "calc" : calc_pars, 
            "final": final_pars
        }
    
    def _analyze_models(self, algo, pre_rotor, models):

        tmodels = []
        tmsels  = []
        mnames  = [[m.name for m in mlist] for mlist in models]
        tmis    = np.zeros(algo.n_turbines, dtype=FC.ITYPE)
        news    = True
        while news:
            news = False

            for ti, mlist in enumerate(models):

                if tmis[ti] < len(mlist):

                    mname  = mnames[ti][tmis[ti]]
                    isnext = True
                    for tj, jnames in enumerate(mnames):
                        if tj != ti and mname in jnames \
                            and tmis[tj] < len(jnames) and jnames[tmis[tj]] != mname:
                            isnext = False
                            break

                    if isnext:

                        m = models[ti][tmis[ti]]
                        tmodels.append(m)

                        tsel = np.zeros((algo.n_states, algo.n_turbines), dtype=bool)
                        for tj, jnames in enumerate(mnames):
                            mi = tmis[tj]
                            if mi < len(jnames) and jnames[mi] == mname:
                                ssel        = algo.farm.turbines[tj].mstates_sel[mi]
                                tsel[:, tj] = True if ssel is None else ssel
                                tmis[tj]   += 1
                        tmsels.append(tsel)

                        news = True
                        break
        
        if pre_rotor:
            self.pre_rotor_models      = FarmDataModelList(tmodels)
            self.pre_rotor_models.name = f"{self.name}_prer"
            mtype = "pre-rotor"
        else:
            self.post_rotor_models      = FarmDataModelList(tmodels)
            self.post_rotor_models.name = f"{self.name}_postr"
            mtype = "post-rotor"
                
        for ti, t in enumerate(algo.farm.turbines):
            if tmis[ti] != len(models[ti]):
                raise ValueError(f"Turbine {ti}, {t.name}: Could not find turbine model order that includes all {mtype} turbine models, missing {t.models[tmis[ti]:]}")

        return [m.name for m in tmodels], tmsels

    def collect_models(self, algo):

        # check turbine models, and find turbine types and pre/post-rotor models:
        self.turbine_types = [ None for t in algo.farm.turbines ]
        prer_models        = [[] for t in algo.farm.turbines ]
        postr_models       = [[] for t in algo.farm.turbines ]
        for ti, t in enumerate(algo.farm.turbines):
            prer = None
            for mi, mname in enumerate(t.models):

                istype = False
                if mname in algo.mbook.turbine_types:
                    m = algo.mbook.turbine_types[mname]
                    if not isinstance(m, TurbineType):
                        raise TypeError(f"Model {mname} type {type(m).__name__} is not derived from {TurbineType.__name__}")
                    istype = True
                elif mname in algo.mbook.turbine_models:
                    m = algo.mbook.turbine_models[mname]
                    if not isinstance(m, TurbineModel):
                        raise TypeError(f"Model {mname} type {type(m).__name__} is not derived from {TurbineModel.__name__}")
                else:
                    raise KeyError(f"Model {mname} not found in model book types or models")
                if istype:
                    if self.turbine_types[ti] is None:
                        self.turbine_types[ti] = m
                    else:
                        raise ValueError(f"Turbine {ti}, {t.name}: Multiple turbine types found in self.turbine_models list, {self.turbine_types[ti].name} and {mname}")

                m.name = mname
                if prer is None:
                    prer = m.pre_rotor
                elif not prer and m.pre_rotor:
                    raise ValueError(f"Turbine {ti}, {t.name}: Model is classified as pre-rotor, but following the post-rotor model '{t.models[mi-1]}'")
                if m.pre_rotor:
                    prer_models[ti].append(m)
                else:
                    postr_models[ti].append(m)

            if self.turbine_types[ti] is None:
                raise ValueError(f"Turbine {ti}, {t.name}: Missing a turbine type model among models {t.models}")

        # analyze models:
        mnames_pre, tmsels_pre   = self._analyze_models(algo, pre_rotor=True, models=prer_models)
        mnames_post, tmsels_post = self._analyze_models(algo, pre_rotor=False, models=postr_models)
        tmsels                   = tmsels_pre + tmsels_post
        self.turbine_model_names = mnames_pre + mnames_post
        if len(self.turbine_model_names):
            self.turbine_model_sels = np.stack(tmsels, axis=2)
        else:
            raise ValueError(f"Controller '{self.name}': No turbine model found.")

    def model_input_data(self, algo):

        if self.turbine_model_names is None:
            self.collect_models(algo)

        idata = super().model_input_data(algo)
        idata["coords"][FV.TMODELS] = self.turbine_model_names
        idata["data_vars"][FV.TMODEL_SELS] = (
            (FV.STATE, FV.TURBINE, FV.TMODELS), self.turbine_model_sels
        )

        return idata

    def output_farm_vars(self, algo):
        if self.turbine_model_names is None:
            self.collect_models(algo)
        return list(dict.fromkeys(
                    self.pre_rotor_models.output_farm_vars(algo) \
                    + self.post_rotor_models.output_farm_vars(algo)
                ))
    
    def get_pars(self, algo, models, ptype, mdata=None, st_sel=None, from_data=True):

        if from_data:
            s = mdata[FV.TMODEL_SELS] 
        else:
            if self.turbine_model_names is None:
                self.collect_models(algo)
            s = self.turbine_model_sels
        if st_sel is not None:
            s = s & st_sel[:, :, None]

        pars = [{"st_sel": s[:, :, self.turbine_model_names.index(m.name)]} for m in models]
        for mi, m in enumerate(models):
            if m.name in self.pars:
                pars[mi].update(self.pars[m.name][ptype])
        
        return pars

    def initialize(self, algo, st_sel=None, verbosity=0):

        if self.turbine_model_names is None:
            self.collect_models(algo)
        
        super().initialize(algo)

        for s in [self.pre_rotor_models, self.post_rotor_models]:
            pars = self.get_pars(algo, s.models, "init", st_sel=st_sel, from_data=False)
            s.initialize(algo, parameters=pars, verbosity=verbosity)
    
    def calculate(self, algo, mdata, fdata, pre_rotor, st_sel=None):
        s    = self.pre_rotor_models if pre_rotor else self.post_rotor_models
        pars = self.get_pars(algo, s.models, "calc", mdata, st_sel, from_data=True)
        res  = s.calculate(algo, mdata, fdata, parameters=pars)
        self.turbine_model_sels = mdata[FV.TMODEL_SELS] 
        return res
    
    def finalize(self, algo, st_sel=None, verbosity=0, clear_mem=False):

        for s in [self.pre_rotor_models, self.post_rotor_models]:
            if s is not None:
                pars = self.get_pars(algo, s.models, "final", st_sel=st_sel, from_data=False)
                s.finalize(algo, parameters=pars, verbosity=verbosity, clear_mem=clear_mem)
        
        super().finalize(algo, clear_mem=clear_mem)
