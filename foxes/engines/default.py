from foxes.core import Engine

class DefaultEngine(Engine):
    """
    The case size dependent default engine.

    :group: engines

    """                  
    def run_calculation(self, algo, *args, **kwargs):
        """
        Runs the model calculation

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm object
        args: tuple, optional
            Additional arguments for the calculation
        kwargs: dict, optional
            Additional arguments for the calculation
            
        Returns
        -------
        results: xarray.Dataset
            The model results

        """
        ename = "single" if (
            (algo.n_states <= 4000 and algo.n_turbines <= 20) or
            (algo.n_states <= 50)
        ) else "process"
        
        self.print(f"{type(self).__name__}: Selecting engine '{ename}'")
        
        self.finalize()
        
        with Engine.new(
            ename, 
            n_procs=self.n_procs, 
            chunk_size_states=self.chunk_size_states,
            chunk_size_points=self.chunk_size_points,
            verbosity=self.verbosity,
        ) as e:
            results = e.run_calculation(algo, *args, **kwargs)
    
        self.initialize()
        
        return results
            