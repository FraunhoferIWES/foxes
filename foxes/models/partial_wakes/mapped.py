from copy import deepcopy

from foxes.core import PartialWakesModel
from foxes.models.partial_wakes.rotor_points import RotorPoints
from foxes.models.partial_wakes.top_hat import PartialTopHat
from foxes.models.partial_wakes.axiwake import PartialAxiwake
from foxes.models.partial_wakes.distsliced import PartialDistSlicedWake
from foxes.models.wake_models.top_hat import TopHatWakeModel
from foxes.models.wake_models.dist_sliced import DistSlicedWakeModel
from foxes.models.wake_models.axisymmetric import AxisymmetricWakeModel
import foxes.constants as FC


class Mapped(PartialWakesModel):
    """
    Partial wake models depending on the wake model (type).

    This is required if more than one wake models are
    used and different partial wake models should be invoked.

    Attributes
    ----------
    wname2pwake: dict
        Mapping from wake model name to partial wakes.
        Key: model name str, value: Tuple of length 2,
        (Partial wake class name, parameter dict)
    wtype2pwake: dict
        Mapping from wake model class name to partial wakes.
        Key: wake model class name str, value: Tuple of length 2,
        (Partial wake class name, parameter dict)

    :group: models.partial_wakes

    """

    def __init__(
        self, wname2pwake={}, wtype2pwake=None, wake_models=None, wake_frame=None
    ):
        """
        Constructor.

        Parameters
        ----------
        wname2pwake: dict, optional
            Mapping from wake model name to partial wakes.
            Key: model name str, value: Tuple of length 2,
            (Partial wake class name, parameter dict)
        wtype2pwake: dict, optional
            Mapping from wake model class name to partial wakes.
            Key: wake model class name str, value: Tuple of length 2,
            (Partial wake class name, parameter dict)
        wake_models: list of foxes.core.WakeModel, optional
            The wake models, default are the ones from the algorithm
        wake_frame: foxes.core.WakeFrame, optional
            The wake frame, default is the one from the algorithm

        """
        super().__init__(wake_models, wake_frame)

        self.wname2pwake = wname2pwake

        if wtype2pwake is None:
            self.wtype2pwake = {
                TopHatWakeModel: (PartialTopHat.__name__, {}),
                AxisymmetricWakeModel: (PartialAxiwake.__name__, {"n": 6}),
                DistSlicedWakeModel: (PartialDistSlicedWake.__name__, {"n": 9}),
            }
        else:
            self.wtype2pwake = wtype2pwake

        self._pwakes = None

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        self.wake_models = algo.wake_models if self._wmodels is None else self._wmodels
        self.wake_frame = algo.wake_frame if self._wframe is None else self._wframe

        pws = {}
        for w in self.wake_models:
            pdat = None
            if w.name in self.wname2pwake:
                pdat = deepcopy(self.wname2pwake[w.name])

            if pdat is None:
                for pwcls, tdat in self.wtype2pwake.items():
                    if isinstance(w, pwcls):
                        pdat = deepcopy(tdat)
                        break

            if pdat is None:
                pdat = (RotorPoints.__name__, {})

            pname = pdat[0]
            if pname not in pws:
                pws[pname] = pdat[1]
                pws[pname]["wake_models"] = []
                pws[pname]["wake_frame"] = self.wake_frame
            pws[pname]["wake_models"].append(w)

        self._pwakes = []
        for pname, pars in pws.items():
            if verbosity:
                print(
                    f"Partial wakes '{self.name}': Applying {pname} to {[w.name for w in pars['wake_models']]}"
                )
            self._pwakes.append(PartialWakesModel.new(pname, **pars))

        super().initialize(algo, verbosity)

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return super().sub_models() + self._pwakes

    def new_wake_deltas(self, algo, mdata, fdata):
        """
        Creates new initial wake deltas, filled
        with zeros.

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
        wake_deltas: dict
            Keys: Variable name str, values: any
        pdata: foxes.core.Data
            The evaluation point data

        """
        wdeltas = []
        pdatas = []
        for pw in self._pwakes:
            w, p = pw.new_wake_deltas(algo, mdata, fdata)
            wdeltas.append(w)
            pdatas.append(p)

        return wdeltas, pdatas

    def contribute_to_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        wake_deltas,
    ):
        """
        Modifies wake deltas by contributions from the
        specified wake source turbines.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data
        states_source_turbine: numpy.ndarray of int
            For each state, one turbine index corresponding
            to the wake causing turbine. Shape: (n_states,)
        wake_deltas: Any
            The wake deltas object created by the
            `new_wake_deltas` function

        """
        for pwi, pw in enumerate(self._pwakes):
            pw.contribute_to_wake_deltas(
                algo, mdata, fdata, pdata[pwi], states_source_turbine, wake_deltas[pwi]
            )

    def evaluate_results(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        wake_deltas,
        states_turbine,
        amb_res=None,
    ):
        """
        Updates the farm data according to the wake
        deltas.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
            Modified in-place by this function
        pdata: foxes.core.Data
            The evaluation point data
        wake_deltas: Any
            The wake deltas object, created by the
            `new_wake_deltas` function and filled
            by `contribute_to_wake_deltas`
        states_turbine: numpy.ndarray of int
            For each state, the index of one turbine
            for which to evaluate the wake deltas.
            Shape: (n_states,)
        amb_res: dict, optional
            Ambient states results. Keys: var str, values:
            numpy.ndarray of shape (n_states, n_points)

        """
        if amb_res is None:
            ares = algo.rotor_model.from_data_or_store(
                FC.AMB_RPOINT_RESULTS, algo, mdata
            ).copy()
            amb_res = {v: d.copy() for v, d in ares.items()}

        for pwi, pw in enumerate(self._pwakes):
            pw.evaluate_results(
                algo,
                mdata,
                fdata,
                pdata[pwi],
                wake_deltas[pwi],
                states_turbine,
                amb_res=amb_res,
            )
