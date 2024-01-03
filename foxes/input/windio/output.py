
from foxes.output import Output

class WIOOutput:
    """
    Output generator initialized from windio data
    
    Attributes
    ----------
    oclass: str
        Name of the output class
    ocargs: dict
        Additional parameters for oclass constructor
    ofunction: str
        The name of the function in the output class
    needs_farm_results: bool
        Flag that farm_results are needed
    name: str
        The output name
    ofargs: dict, optional
        Additional parameters for the output function

    :group: input.windio

    """

    def __init__(self, oclass, ofunction, needs_farm_results=True, 
                 ocargs={}, name="output", **ofargs):
        """
        Constructor.
        
        Parameters
        ----------
        oclass: str
            Name of the output class
        ofunction: str
            The name of the function in the output class
        needs_farm_results: bool
            Flag that farm_results are needed
        ocargs: dict
            Additional parameters for oclass constructor
        name: str
            The output name
        ofargs: dict, optional
            Additional parameters for the output function

        """
        self.oclass = oclass
        self.ocargs = ocargs
        self.ofunction = ofunction
        self.needs_farm_results = needs_farm_results
        self.name = name
        self.ofargs = ofargs
    
    def create(self, *args, farm_results=None, out_dir=None, auto_fnames=False, **kwargs):
        """
        Creates the output
        
        Parameters
        ----------
        args: tuple, optional
            Additional parameters for the output function
        farm_results: xarray.Dataset, optional
            The farm results, only used if flag needs_farm_results
            is set
        out_dir: str, optional
            Sets the output file directory
        auto_fnames: bool or Function
            Flag for automatic file names
        kwargs: dict, optional
            Additional parameters for the output function
        
        Returns
        -------
        output: object
            The output from the output class

        """
        if self.needs_farm_results:
            if farm_results is None:
                raise KeyError("Missing farm_results")
            o = Output.new(self.oclass, farm_results=farm_results, **self.ocargs)
        else:
            o = Output.new(self.oclass, **self.ocargs)

        if out_dir is not None:
            o.out_dir = out_dir
        if isinstance(auto_fnames, bool):
            if auto_fnames:
                o.out_fname_fun = lambda fname: fname.parent/f"{self.name}_{fname.name}"
        else:
            o.out_fname_fun = auto_fnames

        f = getattr(o, self.ofunction)
        return f(*args, **kwargs, **self.ofargs)
