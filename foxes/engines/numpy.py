from xarray import Dataset

from foxes.core import Engine
import foxes.constants as FC 

class NumpyEngine(Engine):
    """
    The numpy engine for foxes calculations.
            
    :group: engines
    
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor.
        
        Parameters
        ----------
        args: tuple, optional
            Additional parameters for the base class
        kwargs: dict, optional
            Additional parameters for the base class
            
        """
        ignr = ["chunk_size_states", "chunk_size_points", "n_procs"]
        for k in ignr:
            if kwargs.pop(k, None) is not None:
                print(f"NumpyEngine: Ignoring {k}")
        super().__init__(
            *args,
            chunk_size_states=None, 
            chunk_size_points=None, 
            **kwargs,
        )
        
    def run_calculation(
        self, 
        algo,
        model, 
        model_data=None, 
        farm_data=None, 
        point_data=None, 
        out_vars=[],
        sel=None,
        isel=None,
        iterative=False,
        **calc_pars,
    ):
        """
        Runs the model calculation
        
        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm object
        model: foxes.core.DataCalcModel
            The model that whose calculate function 
            should be run
        model_data: xarray.Dataset
            The initial model data
        farm_data: xarray.Dataset
            The initial farm data
        point_data: xarray.Dataset
            The initial point data
        out_vars: list of str, optional
            Names of the output variables
        sel: dict, optional
            Selection of coordinate subsets
        isel: dict, optional
            Selection of coordinate subsets index values
        iterative: bool
            Flag for use within the iterative algorithm
        calc_pars: dict, optional
            Additional parameters for the model.calculate()
        
        Returns
        -------
        results: xarray.Dataset
            The model results
            
        """
        # subset selection:
        model_data, farm_data, point_data = self.select_subsets(
            model_data, farm_data, point_data, sel=sel, isel=isel)
        
        # basic checks:
        super().run_calculation(algo, model, model_data, farm_data,
                                point_data, **calc_pars)
        
        # prepare:
        n_states = model_data.sizes[FC.STATE] 
        n_targets = point_data.sizes[FC.TARGET] if point_data is not None else 0
        out_coords = model.output_coords()
        coords = {}
        if FC.STATE in out_coords and FC.STATE in model_data.coords:
            coords[FC.STATE] = model_data[FC.STATE].to_numpy()
        if farm_data is None:
            farm_data = Dataset()
        goal_data = farm_data if point_data is None else point_data
            
        # calculate:
        
        if n_states > 1:
            self.print(f"Running single chunk calculation for {n_states} states")
        
        data = self.get_chunk_input_data(
            algo=algo,
            model_data=model_data, 
            farm_data=farm_data, 
            point_data=point_data, 
            chunki_states=0, 
            chunki_points=0,
            states_i0_i1=(0, n_states),
            targets_i0_i1=(0, n_targets),
            out_vars=out_vars,
            iterative=iterative,
            chunk_store=algo._chunk_store,
        )
        
        results = {}
        results[(0, 0)] = (
            model.calculate(algo, *data, **calc_pars),
            algo._chunk_store
        )

        return self.combine_results(
            algo=algo,
            results=results,
            model_data=model_data,
            out_vars=out_vars,
            out_coords=out_coords,
            n_chunks_states=1,
            n_chunks_targets=1,
            goal_data=goal_data,
            iterative=iterative,
        )
    