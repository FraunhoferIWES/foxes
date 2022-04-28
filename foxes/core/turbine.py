import numpy as np

class Turbine:

    def __init__(
        self,
        xy,
        turbine_models=[],
        index=None,
        label=None,
        models_state_sel=None,
        D=None,
        H=None
    ):
        self.index  = index
        self.label  = label
        self.xy     = np.array(xy)
        self.models = turbine_models
        self.D      = D
        self.H      = H
        
        self.mstates_sel = models_state_sel
        if self.mstates_sel is None:
            self.mstates_sel = [None] * len(self.models)
    
    def add_model(self, model, states_sel=None):
        self.models.append(model)
        self.mstates_sel.append(states_sel)
    
    def insert_model(self, index, model, states_sel=None):
        self.models.insert(index, model)
        self.mstates_sel.insert(index, states_sel)
        