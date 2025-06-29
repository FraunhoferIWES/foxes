from foxes.algorithms.downwind.downwind import Downwind

from foxes.core import FarmDataModelList
from foxes.utils import Dict

from . import models as mdls


class Iterative(Downwind):
    """
    Iterative calculation of farm data.

    Attributes
    ----------
    max_it: int
        The maximal number of iterations
    conv_crit: foxes.algorithms.iterative.ConvCrit
        The convergence criteria
    prev_farm_results: xarray.Dataset
        Results from the previous iteration

    :group: algorithms.iterative

    """

    @classmethod
    def get_model(cls, name):
        """
        Get the algorithm specific model

        Parameters
        ----------
        name: str
            The model name

        Returns
        -------
        model: foxes.core.model
            The model

        """
        try:
            return getattr(mdls, name)
        except AttributeError:
            return super().get_model(name)

    def __init__(
        self,
        *args,
        max_it=None,
        conv_crit="default",
        mod_cutin={},
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Arguments for Downwind
        max_it: int, optional
            The maximal number of iterations
        conv_crit: foxes.algorithms.iterative.ConvCrit, optional
            The convergence criteria
        mod_cutin: dict, optional
            Parameters for cutin modification
        kwargs: dict, optional
            Keyword arguments for Downwind

        """
        super().__init__(*args, **kwargs)

        self.max_it = 2 + self.farm.n_turbines**2 if max_it is None else max_it
        self.conv_crit = (
            self.get_model("DefaultConv")() if conv_crit == "default" else conv_crit
        )
        self.__prev_farm_results = None
        self._it = None
        self._mlist = None
        self._reamb = False
        self._urelax = None

        self._mod_cutin = dict(modify_ct=True, modify_P=False)
        self._mod_cutin.update(mod_cutin)

        self.verbosity = self.verbosity - 1

    @property
    def farm_results_downwind(self):
        """
        Gets the all-chunks farm results in downwind order
        from the previous iteration

        Returns
        -------
        fres: xarray.Datatset
            The all-chunks farm results during calculations

        """
        return self.__prev_farm_results

    def set_urelax(self, entry_point, **urel):
        """
        Sets under-relaxation parameters.

        Parameters
        ----------
        entry_point: str
            The entry point: first, pre_rotor, post_rotor,
            pre_wake, last
        urel: dict
            The variables and their under-relaxation values

        """
        if self.initialized:
            raise ValueError("Attempt to set_urelax after initialization")
        if self._urelax is None:
            self._urelax = Dict(
                first={},
                pre_rotor={},
                post_rotor={},
                pre_wake={},
                last={},
            )
        self._urelax[entry_point].update(urel)

    def initialize(self):
        """
        Initializes the algorithm.
        """
        super().initialize()
        if len(self._mod_cutin):
            for t in self.farm_controller.turbine_types:
                t.modify_cutin(**self._mod_cutin)

    @property
    def urelax(self):
        """
        Returns the under-relaxation parameters

        Returns
        -------
        urlx: foxes.utils.Dict
            The under-relaxation parameters

        """
        return self._urelax

    @property
    def iterations(self):
        """
        The current iteration number

        Returns
        -------
        it: int
            The current iteration number

        """
        return self._it

    def _collect_farm_models(
        self,
        outputs,
        calc_parameters,
        ambient,
    ):
        """
        Helper function that creates model list
        """
        if self._it == 0:
            self._mlist0, self._calc_pars0 = super()._collect_farm_models(
                outputs=False,
                calc_parameters=calc_parameters,
                ambient=ambient,
            )

            n = 0
            if self._urelax is not None:
                if len(self._urelax["first"]):
                    self._mlist0.insert(0, mdls.URelax(**self._urelax["first"]))
                    self._calc_pars0.insert(0, {})
                    self._reamb = True
                    n += 1

                if len(self._urelax["pre_rotor"]):
                    self._mlist0.insert(2 + n, mdls.URelax(**self._urelax["pre_rotor"]))
                    self._calc_pars0.insert(2 + n, {})
                    self._reamb = True
                    n += 1

                if len(self._urelax["post_rotor"]):
                    self._mlist0.insert(
                        4 + n, mdls.URelax(**self._urelax["post_rotor"])
                    )
                    self._calc_pars0.insert(4 + n, {})
                    self._reamb = True
                    n += 1

                if len(self._urelax["pre_wake"]):
                    self._mlist0.models[5 + n].urelax = mdls.URelax(
                        **self._urelax["pre_wake"]
                    )

                if len(self._urelax["last"]):
                    self._mlist0.append(mdls.URelax(**self._urelax["last"]))
                    self._calc_pars0.append({})

            return self._mlist0, self._calc_pars0

        elif ambient or self._reamb:
            return self._mlist0, self._calc_pars0

        else:
            # prepare:
            calc_pars = []
            mlist = FarmDataModelList(models=[])

            # do not rotate back from downwind order:
            if not self._final_run:
                # add under-relaxation during wake calculation:
                urelax = None
                if self._urelax is not None and len(self._urelax["pre_wake"]):
                    urelax = mdls.URelax(**self._urelax["pre_wake"])

                # add model that calculates wake effects:
                mlist.models.append(
                    self.get_model("FarmWakesCalculation")(urelax=urelax)
                )
                calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

                # add under-relaxation:
                if self._urelax is not None and len(self._urelax["last"]):
                    mlist.append(mdls.URelax(**self._urelax["last"]))
                    calc_pars.append({})

            # rotate back from downwind order:
            else:
                mlist.models.append(self.get_model("ReorderFarmOutput")(outputs))
                calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

            return mlist, calc_pars

    def _calc_farm_vars(self, mlist):
        """Helper function that gathers the farm variables"""
        if self._it == 0:
            super()._calc_farm_vars(mlist)

    def _launch_parallel_farm_calc(self, mlist, *data, **kwargs):
        """Helper function for running the main farm calculation"""
        return super()._launch_parallel_farm_calc(
            mlist, *data, farm_data=self.__prev_farm_results, iterative=True, **kwargs
        )

    @property
    def final_iteration(self):
        """
        Flag for the final iteration

        Returns
        -------
        flag: bool
            Flag for the final iteration

        """
        return self._final_run

    def calc_farm(self, finalize=True, ret_dwnd_order=False, **kwargs):
        """
        Calculate farm data.

        Parameters
        ----------
        finalize: bool
            Flag for finalization after calculation
        ret_dwnd_order: bool
            Also return the results in downwind order
        kwargs: dict, optional
            Arguments for calc_farm in the base class.

        Returns
        -------
        farm_results: xarray.Dataset
            The farm results. The calculated variables have
            dimensions (state, turbine)

        """
        outputs = kwargs.pop("outputs", None)
        if outputs == "default":
            outputs = self.DEFAULT_FARM_OUTPUTS

        fres = None
        self._it = -1
        self._final_run = False
        fres_dwnd = None
        while self._it < self.max_it:
            self._it += 1

            self.print(f"\nAlgorithm {self.name}: Iteration {self._it}\n", vlim=0)

            self.__prev_farm_results = fres
            fres = super().calc_farm(outputs=None, finalize=False, **kwargs)

            fres_dwnd = fres
            if self.conv_crit is not None:
                if self.eval_conv_block():
                    self.print(f"{self.name}: Convergence blocked", vlim=0)
                else:
                    conv = self.conv_crit.check_converged(
                        self,
                        self.__prev_farm_results,
                        fres,
                        verbosity=self.verbosity + 1,
                    )
                    if conv:
                        self.print(
                            f"\nAlgorithm {self.name}: Convergence reached.\n", vlim=0
                        )
                        break

        # final run, recovers farm order of results:
        self.print("Starting final run", vlim=0)
        self._final_run = True
        fres = super().calc_farm(outputs=outputs, finalize=False, **kwargs)

        # finalize models:
        if finalize:
            self.print("\n", vlim=0)
            self.finalize()
            for m in self._mlist0.models:
                if m not in self.sub_models():
                    m.finalize(self, self.verbosity - 1)
            del self._mlist0, self._calc_pars0

        if ret_dwnd_order:
            return fres, fres_dwnd
        else:
            return fres
