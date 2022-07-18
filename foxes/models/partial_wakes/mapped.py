from copy import deepcopy

from foxes.core import PartialWakesModel
from foxes.models.partial_wakes.rotor_points import RotorPoints
from foxes.models.partial_wakes.top_hat import PartialTopHat
from foxes.models.partial_wakes.axiwake import PartialAxiwake
from foxes.models.partial_wakes.distsliced import PartialDistSlicedWake
from foxes.models.wake_models.top_hat import TopHatWakeModel
from foxes.models.wake_models.dist_sliced import DistSlicedWakeModel
from foxes.models.wake_models.axisymmetric import AxisymmetricWakeModel


class Mapped(PartialWakesModel):
    """
    Partial wake models depending on the wake model (type).

    This is required if more than one wake models are
    used and different partial wake models should be invoked.

    Parameters
    ----------
    wname2pwake : dict, optional
        Mapping from wake model name to partial wakes.
        Key: model name str, value: Tuple of length 2,
        (Partial wake class name, parameter dict)
    wtype2pwake : dict, optional
        Mapping from wake model class name to partial wakes.
        Key: wake model class name str, value: Tuple of length 2,
        (Partial wake class name, parameter dict)
    wake_models : list of foxes.core.WakeModel, optional
        The wake models, default are the ones from the algorithm
    wake_frame : foxes.core.WakeFrame, optional
        The wake frame, default is the one from the algorithm

    Attributes
    ----------
    wname2pwake : dict
        Mapping from wake model name to partial wakes.
        Key: model name str, value: Tuple of length 2,
        (Partial wake class name, parameter dict)
    wtype2pwake : dict
        Mapping from wake model class name to partial wakes.
        Key: wake model class name str, value: Tuple of length 2,
        (Partial wake class name, parameter dict)

    """

    def __init__(
        self, wname2pwake={}, wtype2pwake=None, wake_models=None, wake_frame=None
    ):
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
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level

        """
        super().initialize(algo, verbosity=verbosity)

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

        for pwake in self._pwakes:
            pwake.initialize(algo, verbosity=verbosity)

    def new_wake_deltas(self, algo, mdata, fdata):
        """
        Creates new initial wake deltas, filled
        with zeros.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data

        Returns
        -------
        wake_deltas : dict
            Keys: Variable name str, values: any

        """
        return [pw.new_wake_deltas(algo, mdata, fdata) for pw in self._pwakes]

    def contribute_to_wake_deltas(
        self, algo, mdata, fdata, states_source_turbine, wake_deltas
    ):
        """
        Modifies wake deltas by contributions from the
        specified wake source turbines.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray of int
            For each state, one turbine index corresponding
            to the wake causing turbine. Shape: (n_states,)
        wake_deltas : Any
            The wake deltas object created by the
            `new_wake_deltas` function

        """
        for pwi, pw in enumerate(self._pwakes):
            pw.contribute_to_wake_deltas(
                algo, mdata, fdata, states_source_turbine, wake_deltas[pwi]
            )

    def evaluate_results(
        self, algo, mdata, fdata, wake_deltas, states_turbine, update_amb_res=True
    ):
        """
        Updates the farm data according to the wake
        deltas.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
            Modified in-place by this function
        wake_deltas : Any
            The wake deltas object, created by the
            `new_wake_deltas` function and filled
            by `contribute_to_wake_deltas`
        states_turbine : numpy.ndarray of int
            For each state, the index of one turbine
            for which to evaluate the wake deltas.
            Shape: (n_states,)
        update_amb_res : bool
            Flag for updating ambient results

        """
        for pwi, pw in enumerate(self._pwakes):
            pw.evaluate_results(
                algo, mdata, fdata, wake_deltas[pwi], states_turbine, update_amb_res
            )
