import numpy as np
import xarray as xr
from multiprocess import Pool
from copy import deepcopy
from os import cpu_count
from tqdm import tqdm

from foxes.core import Engine, MData, FData, TData
import foxes.constants as FC

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
        super().run_calculation(model, model_data, farm_data,
                                point_data, **calc_pars)
        
        # subset selection:
        if sel is not None:
            new_data = []
            for data in [model_data, farm_data, point_data]:
                if data is not None:
                    s = {c: u for c, u in sel.items() if c in data.coords}
                    if len(s):
                        new_data.append(data.sel(s))
                else:
                    new_data.append(data)
            model_data, farm_data, point_data = new_data
            del new_data, s
        if isel is not None:
            new_data = []
            for data in [model_data, farm_data, point_data]:
                if data is not None:
                    s = {c: u for c, u in isel.items() if c in data.coords}
                    if len(s):
                        new_data.append(data.isel(s))
                else:
                    new_data.append(data)
            model_data, farm_data, point_data = new_data
            del new_data, s
            
        # prepare:
        n_states = model_data.sizes[FC.STATE] 
        out_coords = model.output_coords()
        coords = {}
        if FC.STATE in out_coords and FC.STATE in model_data.coords:
            coords[FC.STATE] = model_data[FC.STATE].to_numpy()
        if farm_data is None:
            farm_data = xr.Dataset()

        # determine states chunks:    
        n_chunks_states = 1
        chunk_sizes_states = [n_states]
        n_procs = cpu_count() if self.n_procs is None else self.n_procs
        if self.chunk_size_states is None: 
            chunk_size_states = max(int(n_states/n_procs), 1)
        else: 
            chunk_size_states = min(n_states, self.chunk_size_states)
        n_chunks_states = int(n_states/chunk_size_states)
        chunk_size_states = int(n_states/n_chunks_states)
        chunk_sizes_states = np.full(n_chunks_states, chunk_size_states, dtype=np.int32)
        extra = n_states - n_chunks_states*chunk_size_states
        if extra > 0:
            chunk_sizes_states[-extra:] += 1
        assert np.sum(chunk_sizes_states) == n_states
                
        # determine points chunks:        
        n_targets = point_data.sizes[FC.TARGET] if point_data is not None else 0
        n_chunks_targets = 1
        chunk_sizes_targets = [n_targets]
        if point_data is not None:
            chunk_size_targets = min(n_targets, self.chunk_size_points)
            n_chunks_targets = max(int(n_targets/chunk_size_targets), 1)
            chunk_size_targets = int(n_targets/n_chunks_targets)
            chunk_sizes_targets = np.full(n_chunks_targets, chunk_size_targets, dtype=np.int32)
            extra = n_targets - n_chunks_targets*chunk_size_targets
            if extra > 0:
                chunk_sizes_targets[-extra:] += 1
            assert np.sum(chunk_sizes_targets) == n_targets
                
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
                cpars = deepcopy(calc_pars)
                cpars["algo"] = algo
                
                # create mdata:
                cpars["mdata"] = MData.from_dataset(
                    model_data, s_states=s_states, loop_dims=[FC.STATE], copy=True)
                
                # create fdata:
                if point_data is None:
                    def cb(data, dims):
                        n_states = i1_states - i0_states
                        for o in set(out_vars).difference(data.keys()):
                            data[o] = np.full((n_states, algo.n_turbines), np.nan, dtype=FC.DTYPE)
                            dims[o] = (FC.STATE, FC.TURBINE)
                else:
                    cb = None
                cpars["fdata"] = FData.from_dataset(
                    farm_data, mdata=cpars["mdata"], s_states=s_states, callback=cb,
                    loop_dims=[FC.STATE], copy=True)
            
                # create tdata:
                if point_data is not None:
                    def cb(data, dims):
                        n_states = i1_states - i0_states
                        n_targets = i1_targets - i0_targets
                        for o in set(out_vars).difference(data.keys()):
                            data[o] = np.full((n_states, n_targets, 1), np.nan, dtype=FC.DTYPE)
                            dims[o] = (FC.STATE, FC.TARGET, FC.TPOINT)
                    cpars["tdata"] = TData.from_dataset(
                        point_data, mdata=cpars["mdata"], s_states=s_states, s_targets=s_targets,
                        callback=cb, loop_dims=[FC.STATE, FC.TARGET], copy=True)
                del cb

                # submit model calculation:
                jobs[(chunki_states, chunki_points)] = self._pool.apply_async(
                    model.calculate,
                    kwds=cpars,
                )
                    
                i0_targets = i1_targets
                
                del cpars
                if pbar is not None:
                    pbar.update()
                    
            i0_states = i1_states
            
        del model_data, farm_data, point_data, calc_pars
        if pbar is not None:
            pbar.close()
                
        # awaiting results:
        self.print(f"Computing {n_chunks_all} chunks using {n_procs} processes")
        pbar = tqdm(total=n_chunks_all) if n_states > 1 and self.verbosity > 0 else None
        results = {}
        for chunki_states in range(n_chunks_states):
            for chunki_points in range(n_chunks_targets):
                r = jobs.pop((chunki_states, chunki_points))
                results[(chunki_states, chunki_points)] = r.get()
                if pbar is not None:
                    pbar.update()
        if pbar is not None:
            pbar.close()
        del jobs
            
        # combine results:
        self.print("Combining results", level=2)
        pbar = tqdm(total=len(out_vars)) if self.verbosity > 1 else None
        data_vars = {}
        for v in out_vars:
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
        