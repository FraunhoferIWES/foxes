
from foxes.core.farm_data_model import FarmDataModel

class FarmDataModelList(FarmDataModel):

    def __init__(self, models=[]):
        super().__init__()
        self.models = models

    def input_farm_data(self, algo):
        ddict = super().input_farm_data(algo)
        for m in self.models:
            mdict = m.input_farm_data(algo)
            ddict["coords"].update(mdict["coords"])
            ddict["data_vars"].update(mdict["data_vars"])
        return ddict

    def output_farm_vars(self, algo):
        ovars = []
        for m in self.models:
            ovars += m.output_farm_vars(algo)
        return list(dict.fromkeys(ovars))

    def initialize(self, algo, farm_data, parameters=None, verbosity=0):

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
                m.initialize(algo, farm_data, **parameters[mi])

        super().initialize(algo, farm_data)

    def calculate(self, algo, fdata, parameters=[]):

        if parameters is None:
            parameters = [{}] * len(self.models)
        elif not isinstance(parameters, list):
            raise ValueError(f"{self.name}: Wrong parameters type, expecting list, got {type(parameters).__name__}")
        elif len(parameters) != len(self.models):
            raise ValueError(f"{self.name}: Wrong parameters length, expecting list with {len(self.models)} entries, got {len(parameters)}")

        results = {}
        for mi, m in enumerate(self.models):
            #print("MLIST VARS BEFORE",m.name,list(fdata.keys()))
            mres = m.calculate(algo, fdata, **parameters[mi])
            fdata.update(mres)
            results.update(mres)
        
        return results

    def finalize(self, algo, farm_data, parameters=[], verbosity=0):

        if parameters is None:
            parameters = [{}] * len(self.models)
        elif not isinstance(parameters, list):
            raise ValueError(f"{self.name}: Wrong parameters type, expecting list, got {type(parameters).__name__}")
        elif len(parameters) != len(self.models):
            raise ValueError(f"{self.name}: Wrong parameters length, expecting list with {len(self.models)} entries, got {len(parameters)}")

        for mi, m in enumerate(self.models):
            if verbosity > 0:
                print(f"{self.name}, sub-model '{m.name}': Finalizing")
            m.finalize(algo, farm_data, **parameters[mi])  
        
        self.models = None
