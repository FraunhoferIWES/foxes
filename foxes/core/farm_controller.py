import numpy as np

from foxes.core.farm_data_model_list import FarmDataModelList
from foxes.core.turbine_model import TurbineModel
from foxes.core.turbine_type import TurbineType
import foxes.constants as FC
import foxes.variables as FV

class FarmController(FarmDataModelList):
    
    def __init__(self):
        super().__init__()

        self.turbine_types      = None
        self.turbine_models     = None
        self.turbine_model_sels = None
    
    def collect_models(self, algo):

        self.turbine_types      = [ None for t in algo.farm.turbines ]
        self.turbine_models     = []
        self.turbine_model_sels = []
        tmis = np.zeros(algo.n_turbines, dtype=FC.ITYPE)
        news = True
        while news:
            news = False

            for ti, t in enumerate(algo.farm.turbines):
                if tmis[ti] < len(t.models):

                    mname  = t.models[tmis[ti]]
                    isnext = True
                    for tj, u in enumerate(algo.farm.turbines):
                        if tj != ti and mname in u.models \
                            and tmis[tj] < len(u.models) and u.models[tmis[tj]] != mname:
                            isnext = False
                            break

                    if isnext:

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
                        
                        m.name = mname
                        self.turbine_models.append(m)

                        tsel = np.zeros((algo.n_states, algo.n_turbines), dtype=bool)
                        for tj, u in enumerate(algo.farm.turbines):
                            mi = tmis[tj]
                            if mi < len(u.models) and u.models[mi] == mname:
                                if istype:
                                    if self.turbine_types[tj] is None:
                                        self.turbine_types[tj] = m
                                    else:
                                        raise ValueError(f"Turbine {tj}, {u.label}: Multiple turbine types found in self.turbine_models list, {self.turbine_types[tj].name} and {mname}")
                                ssel        = u.mstates_sel[mi]
                                tsel[:, tj] = True if ssel is None else ssel
                                tmis[tj]   += 1
                        self.turbine_model_sels.append(tsel)

                        news = True
                        break
        
        self.turbine_model_sels = np.stack(self.turbine_model_sels, axis=2)

        for ti, t in enumerate(algo.farm.turbines):
            if self.turbine_types[ti] is None:
                raise ValueError(f"Turbine {ti}, {t.label}: Missing a turbine type model among models {t.models}")
            if tmis[ti] != len(t.models):
                raise ValueError(f"Turbine {ti}, {t.label}: Could not find turbine model order that includes all turbine self.turbine_models, missing {t.models[tmis[ti]:]}")

        self.models = self.turbine_models

    def model_input_data(self, algo):

        if self.turbine_models is None:
            self.collect_models(algo)

        idata = super().model_input_data(algo)
        idata["data_vars"][FV.TMODEL_SELS] = (
            (FV.STATE, FV.TURBINE, FV.TMODELS), self.turbine_model_sels
        )

        return idata

    def output_farm_vars(self, algo):
        if self.turbine_models is None:
            self.collect_models(algo)
        return super().output_farm_vars(algo)
    
    def get_pars(self, algo, mdata=None, st_sel=None, from_data=True):
        if from_data:
            s = mdata[FV.TMODEL_SELS] 
        else:
            if self.turbine_models is None:
                self.collect_models(algo)
            s = self.turbine_model_sels
        if st_sel is not None:
            s = s & st_sel[:, :, None]
        return [{"st_sel": s[:, :, mi]} for mi in range(len(self.models))]

    def initialize(self, algo, st_sel=None, verbosity=0):
        pars = self.get_pars(algo, st_sel=st_sel, from_data=False)
        super().initialize(algo, parameters=pars, verbosity=verbosity)
    
    def calculate(self, algo, mdata, fdata, st_sel=None):
        pars = self.get_pars(algo, mdata, st_sel, from_data=True)
        res  = super().calculate(algo, mdata, fdata, parameters=pars)
        self.turbine_model_sels = mdata[FV.TMODEL_SELS] 
        return res
    
    def finalize(self, algo, st_sel=None, verbosity=0):
        pars = self.get_pars(algo, st_sel=st_sel, from_data=False)
        super().finalize(algo, parameters=pars, verbosity=verbosity)
