import numpy as np
import xarray as xr
from multiprocess import Pool
from copy import deepcopy
from tqdm import tqdm

from foxes.core import Engine, MData, FData, TData
from foxes.utils import Dict
import foxes.constants as FC

def _run_as_proc(algo, model, data, **cpars):
    """ Helper function for running in multiprocessing process """
    results = model.calculate(algo, *data, **cpars)
    chunk_store = algo.reset_chunk_store()
    return results, chunk_store

class MultiprocessEngine(Engine):
    """
    The multiprocessing engine for foxes calculations.
    
    Parameters
    ----------
    n_procs: int
        The number of processes to be used,
        or None for automatic
            
    :group: engines
    
    """
    def __init__(
        self, 
        n_procs=None,
        chunk_size_points=None,
        **kwargs,
    ):
        """
        Constructor.
        
        Parameters
        ----------
        n_procs: int, optional
            The number of processes to be used,
            or None for automatic
        chunk_size_points: int, optional
            The size of a points chunk
        kwargs: dict, optional
            Additional parameters for the base class
            
        """
        csp = chunk_size_points if chunk_size_points is not None else 30000
        super().__init__(chunk_size_points=csp, **kwargs)
        self.n_procs = n_procs
        self._Pool = None

    def initialize(self):
        """
        Initializes the engine.
        """      
        self._pool = Pool(processes=self.n_procs)
        super().initialize()

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
        out_coords = model.output_coords()
        coords = {}
        if FC.STATE in out_coords and FC.STATE in model_data.coords:
            coords[FC.STATE] = model_data[FC.STATE].to_numpy()
        if farm_data is None:
            farm_data = xr.Dataset()
        chunk_store = algo.reset_chunk_store()
        goal_data = farm_data if point_data is None else point_data
            
        # calculate chunk sizes:
        n_targets = point_data.sizes[FC.TARGET] if point_data is not None else 0
        n_procs, chunk_sizes_states, chunk_sizes_targets = self.calc_chunk_sizes(n_states, n_targets)
        n_chunks_states = len(chunk_sizes_states)
        n_chunks_targets = len(chunk_sizes_targets)
        self.print(f"Selecting n_chunks_states = {n_chunks_states}, n_chunks_targets = {n_chunks_targets}", level=2)
                
        # prepare and submit chunks:
        n_chunks_all = n_chunks_states*n_chunks_targets
        n_procs = min(n_procs, n_chunks_all)
        self.print(f"Submitting {n_chunks_all} chunks to {n_procs} processes", level=2)
        pbar = tqdm(total=n_chunks_all) if self.verbosity > 1 else None
        jobs = {}
        i0_states = 0
        for chunki_states in range(n_chunks_states):
            i1_states = i0_states + chunk_sizes_states[chunki_states]
            s_states = np.s_[i0_states:i1_states]
            i0_targets = 0          
            for chunki_points in range(n_chunks_targets):
                i1_targets = i0_targets + chunk_sizes_targets[chunki_points]
                s_targets = np.s_[i0_targets:i1_targets]
                
                # create mdata:
                mdata = MData.from_dataset(
                    model_data, s_states=s_states, loop_dims=[FC.STATE], copy=False)
                
                # create fdata:
                if point_data is None:
                    def cb(data, dims):
                        n_states = i1_states - i0_states
                        for o in set(out_vars).difference(data.keys()):
                            data[o] = np.full((n_states, algo.n_turbines), np.nan, dtype=FC.DTYPE)
                            dims[o] = (FC.STATE, FC.TURBINE)
                else:
                    cb = None
                fdata = FData.from_dataset(
                    farm_data, mdata=mdata, s_states=s_states, callback=cb,
                    loop_dims=[FC.STATE], copy=False)
            
                # create tdata:
                if point_data is not None:
                    def cb(data, dims):
                        n_states = i1_states - i0_states
                        n_targets = i1_targets - i0_targets
                        for o in set(out_vars).difference(data.keys()):
                            data[o] = np.full((n_states, n_targets, 1), np.nan, dtype=FC.DTYPE)
                            dims[o] = (FC.STATE, FC.TARGET, FC.TPOINT)
                    tdata = TData.from_dataset(
                        point_data, mdata=mdata, s_states=s_states, s_targets=s_targets,
                        callback=cb, loop_dims=[FC.STATE, FC.TARGET], copy=False)
                else:
                    tdata = None
                del cb
                
                # set chunk store data:
                key = (chunki_states, chunki_points)
                algo._chunk_store = Dict(name=chunk_store.name)
                if key in chunk_store:
                    algo._chunk_store[key] = chunk_store.pop(key)

                # submit model calculation:
                data = [d for d in [mdata, fdata, tdata] if d is not None]
                jobs[key] = self._pool.apply_async(
                    _run_as_proc, 
                    args=(algo, model, data), 
                    kwds=calc_pars,
                )
                del data, mdata, fdata, tdata
                    
                i0_targets = i1_targets
                
                if pbar is not None:
                    pbar.update()
                    
            i0_states = i1_states
            
        del model_data, calc_pars, chunk_store, farm_data, point_data
        if pbar is not None:
            pbar.close()
                
        # wait for results:
        if n_chunks_all > 1 or self.verbosity > 1:
            self.print(f"Computing {n_chunks_all} chunks using {n_procs} processes")
        pbar = tqdm(total=n_chunks_all) if n_chunks_all > 1 and self.verbosity > 0 else None
        results = {}
        for chunki_states in range(n_chunks_states):
            for chunki_points in range(n_chunks_targets):
                key = (chunki_states, chunki_points)
                results[key], cstore = jobs.pop((chunki_states, chunki_points)).get()
                algo._chunk_store.update(cstore)
                if pbar is not None:
                    pbar.update()
        if pbar is not None:
            pbar.close()
        del jobs, cstore
            
        # combine results:
        self.print("Combining results", level=2)
        pbar = tqdm(total=len(out_vars)) if self.verbosity > 1 else None
        data_vars = {}
        for v in out_vars:
            if v in results[(0, 0)]:
                data_vars[v] = [out_coords, []]
                
                if n_chunks_targets == 1:
                    alls=0
                    for chunki_states in range(n_chunks_states):
                        r = results[(chunki_states, 0)]
                        data_vars[v][1].append(r[v])
                        alls += data_vars[v][1][-1].shape[0]
                else:
                    for chunki_states in range(n_chunks_states):
                        tres = []
                        for chunki_points in range(n_chunks_targets):
                            r = results[(chunki_states, chunki_points)]
                            tres.append(r[v])
                        data_vars[v][1].append(np.concatenate(tres, axis=1))
                    del tres
                data_vars[v][1] = np.concatenate(data_vars[v][1], axis=0)
            else:
                data_vars[v] = (goal_data[v].dims, goal_data[v].to_numpy())
            
            if pbar is not None:
                pbar.update()
        del results
        if pbar is not None:
            pbar.close()

        return xr.Dataset(
            coords=coords, 
            data_vars={v: tuple(d) for v, d in data_vars.items()},
        )
        
    def finalize(self):
        """
        Finalizes the engine.
        """
        if self._pool is not None:
            self._pool.close()
            self._pool.terminate()
            self._pool = None
 
        super().finalize()
        