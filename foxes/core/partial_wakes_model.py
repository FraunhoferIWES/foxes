from abc import abstractmethod
import numpy as np

from foxes.utils import new_instance, wd2uv, uv2wd
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

from .model import Model
from .data import TData

class PartialWakesModel(Model):
    """
    Abstract base class for partial wakes models.

    Partial wakes models compute wake effects
    for rotor effective quantities.

    Attributes
    ----------
    wake_models: list of foxes.core.WakeModel
        The wake model selection
    wake_frame: foxes.core.WakeFrame, optional
        The wake frame

    :group: core

    """

    def check_wmodel(self, wmodel, error=True):
        """
        Checks the wake model type

        Parameters
        ----------
        wmodel: foxes.core.WakeModel
            The wake model to be tested
        error: bool
            Flag for raising TypeError

        Returns
        -------
        chk: bool
            True if wake model is compatible

        """
        return True

    @abstractmethod
    def get_wake_points(self, algo, mdata, fdata):
        """
        Get the wake calculation points, and their
        weights.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data

        Returns
        -------
        rpoints: numpy.ndarray
            The wake calculation points, shape:
            (n_states, n_turbines, n_tpoints, 3)
        rweights: numpy.ndarray
            The target point weights, shape: (n_tpoints,)

        """
        pass

    def get_initial_tdata(self, algo, mdata, fdata, amb_res):
        """
        Creates the initial target data object

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        amb_res: dict
            The ambient results at rotor points,
            key: variable name, value: numpy.ndarray
            of shape: (n_states, n_turbines, n_rotor_points)

        Returns
        -------
        tdata: foxes.core.TData
            The target point data for the wake points

        """
        tpoints, tweights = self.get_wake_points(algo, mdata, fdata)
        tdata = TData.from_tpoints(tpoints, tweights)

        # translate WS, WD to UV data:
        if FV.WD in amb_res or FV.WS in amb_res:
            assert FV.WD in amb_res and FV.WS in amb_res, \
                "Require both wind direction and speed in ambient results."
            uv = wd2uv(amb_res.pop(FV.WD), amb_res.pop(FV.WS))
            amb_res[FV.U] = uv[..., 0]
            amb_res[FV.V] = uv[..., 1]

        # map rotor point results onto target points:
        for v, d in amb_res.items():
            w = FV.var2amb.get(v, v)
            print("PWKS ADDING",v,w,d.shape,(tdata.n_states, tdata.n_targets, tdata.n_tpoints), self.map_rotor_results)
            tdata.add(
                w, 
                self.map_rotor_results(algo, mdata, fdata, tdata, v, d), 
                dims=(FC.STATE, FC.TARGET, FC.TPOINT),
            ) 
        print("HERE PWKS",{v:d.shape for v, d in tdata.items()})
        quit()
        
        # back translate UV to WD, WS:
        if FV.AMB_U in tdata:
            uv = np.stack([tdata.pop(FV.AMB_U), tdata.pop(FV.AMB_V)], axis=-1)
            tdata.add(FV.AMB_WD, uv2wd(uv), dims=(FC.STATE, FC.TARGET, FC.TPOINT))
            tdata.add(FV.AMB_WS, np.linalg.norm(uv, axis=-1), dims=(FC.STATE, FC.TARGET, FC.TPOINT))

        print("HERE PWKS",{v:d.shape for v, d in tdata.items()})

        return tdata
    
    def map_rotor_results(self, algo, mdata, fdata, tdata, variable, rotor_res):
        """
        Map ambient rotor point results onto target points.
        
        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        variable: str
            The variable name to map
        rotor_res: numpy.ndarray
            The results at rotor points, shape: 
            (n_states, n_turbines, n_rotor_points)
        
        Returns
        -------
        res: numpy.ndarray
            The mapped results at target points, shape:
            (n_states, n_targets, n_tpoints)

        """
        if len(rotor_res.shape) > 2 and rotor_res.shape[:2] == (tdata.n_states, tdata.n_targets):
            q = np.zeros((tdata.n_states, tdata.n_targets, tdata.n_tpoints), dtype=config.dtype_double)
            if rotor_res.shape[2] == 1:
                q[:] = rotor_res
            else:
                print("HERE PWKS", variable, rotor_res.shape, tdata[FC.TWEIGHTS].shape)
                q[:] = np.einsum('str,r->st', rotor_res, tdata[FC.TWEIGHTS])[:, :, None]
            return q
        else:
            raise ValueError(f"Partial wakes '{self.name}': Incompatible shape '{rotor_res.shape}' for variable '{variable}' in rotor results.")
        
    def new_wake_deltas(self, algo, mdata, fdata, tdata, wmodel):
        """
        Creates new initial wake deltas, filled
        with zeros.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        wmodel: foxes.core.WakeModel
            The wake model

        Returns
        -------
        wake_deltas: dict
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_turbines, n_tpoints, ...)

        """
        return wmodel.new_wake_deltas(algo, mdata, fdata, tdata)

    def contribute(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        wake_deltas,
        wmodel,
    ):
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        downwind_index: int
            The index of the wake causing turbine
            in the downwind order
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model
            
        """
        wcoos, delwd = algo.wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)
        wmodel.contribute(algo, mdata, fdata, tdata, downwind_index, wcoos, delwd, wake_deltas)

    @abstractmethod
    def finalize_wakes(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        amb_res,
        rpoint_weights,
        wake_deltas,
        wmodel,
        downwind_index,
    ):
        """
        Updates the wake_deltas at the selected target
        downwind index.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.Data
            The target point data
        amb_res: dict
            The ambient results at the target points
            of all rotors. Key: variable name, value
            np.ndarray of shape:
            (n_states, n_turbines, n_rotor_points)
        rpoint_weights: numpy.ndarray
            The rotor point weights, shape: (n_rotor_points,)
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: np.ndarray of shape
            (n_states, n_turbines, n_tpoints)
        wmodel: foxes.core.WakeModel
            The wake model
        downwind_index: int
            The index in the downwind order

        Returns
        -------
        final_wake_deltas: dict
            The final wake deltas at the selected downwind
            turbines. Key: variable name, value: np.ndarray
            of shape (n_states, n_rotor_points)

        """
        pass

    @classmethod
    def new(cls, pwakes_type, *args, **kwargs):
        """
        Run-time partial wakes model factory.

        Parameters
        ----------
        pwakes_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for the constructor
        kwargs: dict, optional
            Additional parameters for the constructor

        """
        return new_instance(cls, pwakes_type, *args, **kwargs)
