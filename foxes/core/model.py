import numpy as np
from abc import ABCMeta
from itertools import count

class Model(metaclass=ABCMeta):

    _ids = {}

    def __init__(self):

        t = type(self).__name__
        if t not in self._ids:
            self._ids[t] = count(0)

        self.id   = next(self._ids[t]) 
        self.name = type(self).__name__ 
        self.__initialized = False

    def __repr__(self):
        t = type(self).__name__
        s = self.name if self.name == t else f"{self.name} ({t})"
        return s
    
    def var(self, v):
        return f"{self.name}_{self.id}_{v}"

    def model_input_data(self, algo):
        return {"coords": {}, "data_vars": {}}

    def initialize(self, algo):
        idata = self.model_input_data(algo)
        algo.models_idata["coords"].update(idata["coords"])
        algo.models_idata["data_vars"].update(idata["data_vars"])
        self.__initialized = True
    
    @property
    def initialized(self):
        return self.__initialized

    def get_data(
            self,
            variable, 
            data, 
            st_sel=None, 
            upcast=None, 
            data_prio=True
        ):

        sources = (data, self.__dict__) if data_prio \
                    else (self.__dict__, data)

        try:
            out = sources[0][variable]
        except KeyError:
            try:
                out = sources[1][variable]
            except KeyError:
                raise KeyError(f"Model '{self.name}': Variable '{variable}' neither found in data {sorted(list(data.keys()))} nor among attributes")
        
        if upcast is not None and not isinstance(out, np.ndarray):
            if upcast == "farm":
                out = np.full((data.n_states, data.n_turbines), out)
            elif upcast == "points":
                out = np.full((data.n_states, data.n_points), out)
            else:
                raise ValueError(f"Model '{self.name}': Illegal upcast '{upcast}', select 'farm' or 'points'")

        if st_sel is not None:
            try:
                out = out[st_sel]
            except TypeError:
                pass
        
        return out
        