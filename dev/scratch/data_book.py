from importlib import resources
from pathlib import Path

from . import farms
from . import states
from . import power_ct_curves

FARM     = "farm"
STATES   = "states"
PCTCURVE = "power_ct_curve"

class DataBook:
    
    def __init__(self, add_defaults=True):

        self.dbase = {}

        if add_defaults:
            self.add_data_package(FARM, farms, ".csv")
            self.add_data_package(PCTCURVE, power_ct_curves, ".csv")
            self.add_data_package(STATES, states, [".csv.gz", ".nc"])
    
    def add_data_package(self, context, package, file_sfx):

        if context not in self.dbase:
            self.dbase[context] = {}
        
        if isinstance(file_sfx, str):
            file_sfx = [file_sfx]

        contents = resources.contents(package)
        check_f  = lambda f: any([len(f) > len(s) and f[-len(s):] == s for s in file_sfx])
        contents = [f for f in contents if check_f(f)]

        for f in contents:
            with resources.path(package, f) as path:
                self.dbase[context][f] = path

    def add_data_package_file(self, context, package, file_name):

        if context not in self.dbase:
            self.dbase[context] = {}
        
        try:
            with resources.path(package, file_name) as path:
                self.dbase[context][file_name] = path
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_name}' not found in package '{package}'")

    def add_files(self, context, file_paths):

        if context not in self.dbase:
            self.dbase[context] = {}
        
        for f in file_paths:
            path = Path(f)
            if not path.is_file():
                raise FileNotFoundError(f"File '{path}' not found, cannot add to context '{context}'")
            self.dbase[context][path.name] = path
    
    def add_file(self, context, file_path):
        self.add_files(context, [file_path])

    def get_file_path(self, context, file_name, errors=True):

        try:
            cdata = self.dbase[context]
        except KeyError:
            if not errors:
                return None
            raise KeyError(f"Context '{context}' not found in data book. Available: {sorted(list(self.dbase.keys()))}")
        
        try: 
            return cdata[file_name]
        except KeyError:
            if not errors:
                return None
            raise KeyError(f"File '{file_name}' not found in context '{context}'. Available: {sorted(list(cdata.keys()))}")

