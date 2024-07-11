import numpy as np
import xarray as xr
from multiprocess import Pool
from copy import deepcopy
from os import cpu_count
from tqdm import tqdm

from foxes.core import Engine, MData, FData, TData
import foxes.variables as FV
import foxes.constants as FC

class MultiprocessingEngine(Engine):
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
        **kwargs,
    ):
        """
        Constructor.
        
        Parameters
        ----------
        n_procs: int, optional
            The number of processes to be used,
            or None for automatic
        kwargs: dict, optional
            Additional parameters for the base class
            
        """
        super().__init__(**kwargs)
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
        
        # prepare:
        n_states = model_data.sizes[FC.STATE] 
        out_coords = model.output_coords()
        if out_coords == (FC.STATE, FC.TARGET, FC.TPOINT):
            out_coords = (FC.STATE, FC.POINT)
        coords = {}
        if FC.STATE in out_coords and FC.STATE in model_data.coords:
            coords[FC.STATE] = model_data[FC.STATE].to_numpy()
            
        # determine states chunks:    
        n_chunks_states = 1
        chunk_sizes_states = [n_states]
        if self.chunk_size_states is not None: 
            n_chunks_states = int(n_states/self.chunk_size_states)
            chunk_sizes_states = np.full(n_chunks_states, self.chunk_size_states, dtype=np.int32)
            missing = n_states - n_chunks_states*self.chunk_size_states
            if missing > 0:
                chunk_sizes_states[-missing:] += 1
                
        # determine points chunks:        
        n_targets = point_data.sizes[FC.TARGET] if point_data is not None else 0
        n_chunks_targets = 1
        chunk_sizes_targets = [n_targets]
        if point_data is not None and self.chunk_size_points is not None:
            n_chunks_targets = int(n_targets/self.chunk_size_points)
            chunk_sizes_targets = np.full(n_chunks_targets, self.chunk_size_points, dtype=np.int32)
            missing = n_targets - n_chunks_targets*self.chunk_size_points
            if missing > 0:
                chunk_sizes_targets[-missing:] += 1
        
        # prepare and submit chunks:
        n_procs = cpu_count() if self.n_procs is None else self.n_procs
        n_chunks_all = n_chunks_states*n_chunks_targets
        self.print(f"Submitting {n_chunks_all} chunks to {n_procs} processes", level=2)
        pbar = tqdm(total=n_chunks_all) if self.verbosity > 1 else None
        results = [[[]]*n_chunks_targets]*n_chunks_states
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
                for data, dname, cls in zip(
                    [model_data, farm_data, point_data], 
                    ["mdata", "fdata", "tdata"], 
                    [MData, FData, TData]
                ):
                    if data is not None:
                        dats={}
                        dims={}
                        ldims = set([FC.STATE])
                          
                        # extract coordinates:
                        for c, d in data.coords.items():
                            if c == FC.STATE:
                                dats[c] = d.to_numpy()[s_states].copy()
                            elif c == FC.TARGET:
                                dats[c] = d.to_numpy()[s_targets].copy()
                            else:
                                dats[c] = d.to_numpy().copy()
                            dims[c] = d.dims
                        if FC.STATE not in dats:
                            dats[FC.STATE] = np.arange(i0_states, i1_states)
                            dats[FC.STATE] = (FC.STATE,)
                        if dname == "tdata" and FC.TARGET not in dats:
                            dats[FC.TARGET] = np.arange(i0_targets, i1_targets)
                            dats[FC.TARGET] = (FC.TARGET,)
                        
                        # extract data vars:
                        for v, d in data.data_vars.items():
                            dats[v] = d.to_numpy().copy()
                            dims[v] = d.dims
                            if dname == "tdata" and FC.TARGET in d.dims:
                                assert len(d.dims) > 1 and d.dims[:2] == (FC.STATE, FC.TARGET)
                                ldims.add(FC.TARGET)
                                dats[v] = dats[v][s_states, s_targets] 
                            elif FC.STATE in d.dims:
                                assert d.dims[0] == FC.STATE
                                dats[v] = dats[v][s_states]    
                    
                        ldims = [l for l in self.loop_dims if l in ldims]
                        cpars[dname] = cls(dats, dims, loop_dims=ldims, name=dname)
                        del dats, dims
                    
                    elif dname == "fdata":
                        cpars[dname] = cls({}, {}, loop_dims=ldims, name=dname)
                        
                # link basic data from mdata to fdata and tdata:
                data = [cpars[n] for n in ["mdata", "fdata", "tdata"] if n in cpars]
                if FV.WEIGHT in data[0]:
                    data[1].add(FV.WEIGHT, data[0][FV.WEIGHT], data[0].dims[FV.WEIGHT])
                if FC.STATE in data[0]:
                    for d in data[1:]:
                        d.add(FC.STATE, data[0][FC.STATE], data[0].dims[FC.STATE])

                # submit model calculation:
                model.ensure_variables(algo, *data)
                results[chunki_states][chunki_points] = self._pool.apply_async(
                    model.calculate,
                    kwds=cpars,
                )
                    
                i0_targets = i1_targets
                
                del cpars, data
                if pbar is not None:
                    pbar.update()
                    
            i0_states = i1_states
            
        del model_data, farm_data, point_data, calc_pars
        if pbar is not None:
            pbar.close()
                
        # receive results:
        self.print(f"Computing {n_chunks_all} chunks using {n_procs} processes")
        pbar = tqdm(total=n_chunks_all) if self.verbosity > 0 else None
        for chunki_states in range(n_chunks_states):
            for chunki_points in range(n_chunks_targets):
                r = results[chunki_states][chunki_points]
                results[chunki_states][chunki_points] = r if isinstance(r, dict) else r.get()
                if pbar is not None:
                    pbar.update()
        if pbar is not None:
            pbar.close()
            
        # combine results:
        self.print("Combining results", level=2)
        pbar = tqdm(total=len(out_vars)) if self.verbosity > 1 else None
        data_vars = {}
        for v in out_vars:
            data_vars[v] = [out_coords, []]
            
            if n_chunks_targets == 1:
                for chunki_states in range(n_chunks_states):
                    data_vars[v][1].append(results[chunki_states][0][v])
                    
            else:
                for chunki_states in range(n_chunks_states):
                    data_vars[v][1].append([[]])
                    for chunki_points in range(n_chunks_targets):
                        data_vars[v][1][-1].append(results[chunki_states][chunki_points][v][:, :, 0])
                    data_vars[v][1][-1] = np.concatenate(data_vars[v][1][-1], axis=1)
            data_vars[v][1] = np.concatenate(data_vars[v][1], axis=0)
            
            if pbar is not None:
                pbar.update()
        for v in out_vars:
            data_vars[v] = tuple(data_vars[v])
        del results
        if pbar is not None:
            pbar.close()

        return xr.Dataset(coords=coords, data_vars=data_vars)
        
    def finalize(self):
        """
        Finalizes the engine.
        """
        if self._pool is not None:
            self._pool.close()
            self._pool.terminate()
            self._pool = None
 
        super().finalize()
        