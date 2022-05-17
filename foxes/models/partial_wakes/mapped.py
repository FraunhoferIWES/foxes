import numpy as np
from copy import deepcopy

from foxes.core import PartialWakesModel
from foxes.models.partial_wakes.rotor_points import RotorPoints
from foxes.models.partial_wakes.partial_top_hat import PartialTopHat
from foxes.models.partial_wakes.partial_axiwake import PartialAxiwake
from foxes.models.partial_wakes.partial_distsliced import PartialDistSlicedWake
from foxes.models.wake_models.dist_sliced.axisymmetric.top_hat.top_hat_wake_model import TopHatWakeModel
from foxes.models.wake_models.dist_sliced.dist_sliced_wake_model import DistSlicedWakeModel
from foxes.models.wake_models.dist_sliced.axisymmetric.axisymmetric_wake_model import AxisymmetricWakeModel
import foxes.variables as FV

class Mapped(PartialWakesModel):

    def __init__(self, wname2pwake={}, wtype2pwake=None, wake_models=None, wake_frame=None):
        super().__init__(wake_models, wake_frame)

        self.wname2pwake = wname2pwake
        
        if wtype2pwake is None:
            self.wtype2pwake = {
                TopHatWakeModel      : (PartialTopHat.__name__, {}),
                AxisymmetricWakeModel: (PartialAxiwake.__name__, {"n": 5}),
                DistSlicedWakeModel  : (PartialDistSlicedWake.__name__, {"n": 9})
            }
        else:
            self.wtype2pwake = wtype2pwake

        self._pwakes = None

    def initialize(self, algo):

        super().initialize(algo)

        pws = {}
        for w in self.wake_models:

            pdat = None
            if w.name in self.wname2pwake:
                pdat = deepcopy(self.wname2pwake[w.name])
            
            if pdat is None:
                for pwcls, tdat in self.wtype2pwake.items():
                    if isinstance(w, pwcls):
                        pdat = deepcopy(tdat)
                        break
            
            if pdat is None:
                pdat = (RotorPoints.__name__, {})

            pname = pdat[0]
            if pname not in  pws:
                pws[pname] = pdat[1]
                pws[pname]["wake_models"] = []
                pws[pname]["wake_frame"]  = self.wake_frame
            pws[pname]["wake_models"].append(w)
        
        self._pwakes = []
        for pname, pars in pws.items():
            self._pwakes.append(PartialWakesModel.new(pname, **pars))
            self._pwakes[-1].initialize(algo)

    def new_wake_deltas(self, algo, mdata, fdata):
        return [pw.new_wake_deltas(algo, mdata, fdata) for pw in self._pwakes]

    def contribute_to_wake_deltas(self, algo, mdata, fdata, states_source_turbine, 
                                    wake_deltas):
        
        for pwi, pw in enumerate(self._pwakes):
            pw.contribute_to_wake_deltas(algo, mdata, fdata, states_source_turbine, 
                                    wake_deltas[pwi])

    def evaluate_results(self, algo, mdata, fdata, wake_deltas, states_turbine, update_amb_res=True):
        
        for pwi, pw in enumerate(self._pwakes):
            pw.evaluate_results(algo, mdata, fdata, wake_deltas[pwi], states_turbine, update_amb_res)
                      