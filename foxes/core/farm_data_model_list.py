
from foxes.core.farm_data_model import FarmDataModel

class FarmDataModelList(FarmDataModel):

    def __init__(self, models=[]):
        super().__init__()
        self.models = models

    def output_farm_vars(self, algo):
        ovars = []
        for m in self.models:
            ovars += m.output_farm_vars(algo)
        return list(dict.fromkeys(ovars))

    def initialize(self, algo, parameters=None, verbosity=0):

        if parameters is None:
            parameters = [{}] * len(self.models)
        elif not isinstance(parameters, list):
            raise ValueError(f"{self.name}: Wrong parameters type, expecting list, got {type(parameters).__name__}")
        elif len(parameters) != len(self.models):
            raise ValueError(f"{self.name}: Wrong parameters length, expecting list with {len(self.models)} entries, got {len(parameters)}")

        for mi, m in enumerate(self.models):
            if not m.initialized:
                if verbosity > 0:
                    print(f"{self.name}, sub-model '{m.name}': Initializing")
                m.initialize(algo, **parameters[mi])

        super().initialize(algo)

    def calculate(self, algo, mdata, fdata, parameters=[]):

        if parameters is None:
            parameters = [{}] * len(self.models)
        elif not isinstance(parameters, list):
            raise ValueError(f"{self.name}: Wrong parameters type, expecting list, got {type(parameters).__name__}")
        elif len(parameters) != len(self.models):
            raise ValueError(f"{self.name}: Wrong parameters length, expecting list with {len(self.models)} entries, got {len(parameters)}")

        for mi, m in enumerate(self.models):
            #print("MLIST VARS BEFORE",m.name,list(fdata.keys()),parameters[mi])
            res = m.calculate(algo, mdata, fdata, **parameters[mi])
            fdata.update(res)
        
        return {v: fdata[v] for v in self.output_farm_vars(algo)}

    def finalize(self, algo, parameters=[], verbosity=0, clear_mem=False):

        if parameters is None:
            parameters = [{}] * len(self.models)
        elif not isinstance(parameters, list):
            raise ValueError(f"{self.name}: Wrong parameters type, expecting list, got {type(parameters).__name__}")
        elif len(parameters) != len(self.models):
            raise ValueError(f"{self.name}: Wrong parameters length, expecting list with {len(self.models)} entries, got {len(parameters)}")

        for mi, m in enumerate(self.models):
            if verbosity > 0:
                print(f"{self.name}, sub-model '{m.name}': Finalizing")
            m.finalize(algo, **parameters[mi])  
        
        if clear_mem:
            self.models = None
            
        super().finalize(algo, clear_mem)

