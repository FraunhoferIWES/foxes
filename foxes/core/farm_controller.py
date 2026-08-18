from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from typing import TYPE_CHECKING, Any, cast

from foxes.config import config
import foxes.constants as FC
from foxes.utils import new_instance

from .farm_data_model import FarmDataModelList, FarmDataModel
from .turbine_model import TurbineModel
from .turbine_type import TurbineType

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData
    from foxes.core.model import LoadedData


class FarmController(FarmDataModel):
    """
    Analyses selected turbine models and handles their call.

    Attributes
    ----------
    turbine_types
        The turbine type of each turbine
    turbine_model_names
        Names of all turbine models found in the farm
    pre_rotor_models
        The turbine models with pre-rotor flag
    post_rotor_models
        The turbine models without pre-rotor flag
    pars
        Parameters for the turbine models, stored
        under their respecitve name


    """

    def __init__(
        self, pars: dict[str, dict[str, dict[str, Any]]] | None = None
    ) -> None:
        """
        Construct the farm controller.

        Parameters
        ----------
        pars
            Parameters for the turbine models, stored under their respective
            names.

        """
        super().__init__()

        self.turbine_types: list[TurbineType] | None = None
        self.turbine_model_names: list[str] | None = None
        self.pre_rotor_models: FarmDataModelList | None = None
        self.post_rotor_models: FarmDataModelList | None = None
        self.pars = {} if pars is None else pars
        self._tmall: list[bool] | None = None
        self._tmsels: dict[int, np.ndarray] | None = None

    def sub_models(self) -> list[Any]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            Names of all sub models

        """
        return [
            self.pre_rotor_models,
            self.post_rotor_models,
        ]

    def set_pars(
        self,
        model_name: str,
        init_pars: dict[str, Any],
        calc_pars: dict[str, Any],
        final_pars: dict[str, Any],
    ) -> None:
        """
        Set parameters for a turbine model.

        Parameters
        ----------
        model_name
            Name of the model.
        init_pars
            Parameters for initialization.
        calc_pars
            Parameters for calculation.
        final_pars
            Parameters for finalization.

        """
        self.pars[model_name] = {
            "init": init_pars,
            "calc": calc_pars,
            "final": final_pars,
        }

    def needs_rews2(self) -> bool:
        """
        Returns flag for requiring REWS2 variable

        Returns
        -------
        needs_rews2
            True if REWS2 is required

        """
        assert self.turbine_types is not None
        for tt in self.turbine_types:
            if tt.needs_rews2():
                return True
        return False

    def needs_rews3(self) -> bool:
        """
        Returns flag for requiring REWS3 variable

        Returns
        -------
        needs_rews3
            True if REWS3 is required

        """
        assert self.turbine_types is not None
        for tt in self.turbine_types:
            if tt.needs_rews3():
                return True
        return False

    def _analyze_models(
        self,
        algo: Algorithm,
        pre_rotor: bool,
        models: list[list[Any]],
    ) -> tuple[list[str], list[np.ndarray]]:
        """
        Helper function for model analysis
        """
        tmodels: list[Any] = []
        tmsels: list[np.ndarray] = []
        mnames = [[m.name for m in mlist] for mlist in models]
        tmis: np.ndarray = np.zeros(algo.n_turbines, dtype=config.dtype_int)
        news = True
        while news:
            news = False

            for ti, mlist in enumerate(models):
                if tmis[ti] < len(mlist):
                    mname = mnames[ti][tmis[ti]]
                    isnext = True
                    for tj, jnames in enumerate(mnames):
                        if (
                            tj != ti
                            and mname in jnames
                            and tmis[tj] < len(jnames)
                            and jnames[tmis[tj]] != mname
                        ):
                            isnext = False
                            break

                    if isnext:
                        m = models[ti][tmis[ti]]
                        tmodels.append(m)

                        tsel: np.ndarray = np.zeros(
                            (algo.n_states, algo.n_turbines), dtype=bool
                        )
                        for tj, jnames in enumerate(mnames):
                            mi = tmis[tj]
                            if mi < len(jnames) and jnames[mi] == mname:
                                ssel = algo.farm.turbines[tj].mstates_sel[mi]
                                tsel[:, tj] = True if ssel is None else ssel
                                tmis[tj] += 1
                        tmsels.append(tsel)

                        news = True
                        break

        if pre_rotor:
            self.pre_rotor_models = FarmDataModelList(models=tmodels)
            self.pre_rotor_models.name = f"{self.name}_prer"
            mtype = "pre-rotor"
        else:
            self.post_rotor_models = FarmDataModelList(models=tmodels)
            self.post_rotor_models.name = f"{self.name}_postr"
            mtype = "post-rotor"

        for ti, t in enumerate(algo.farm.turbines):
            if tmis[ti] != len(models[ti]):
                raise ValueError(
                    f"Turbine {ti}, {t.name}: Could not find turbine model order that includes all {mtype} turbine models, missing {t.models[tmis[ti] :]}"
                )

        return [m.name for m in tmodels], tmsels

    def _tmodel_sels_var(self, mi: int) -> str:
        """
        Gets the mdata variable name of turbine model selections.

        Parameters
        ----------
        mi
            The turbine model index

        Returns
        -------
        model_selection_variable
            The per-model selection variable name

        """
        assert self.turbine_model_names is not None
        return self.var("tsel_" + self.turbine_model_names[mi])

    @property
    def has_pre_rotor_models(self) -> bool:
        """
        Flag for having pre-rotor models

        Returns
        -------
        has_pre_rotor_models
            True if pre-rotor models are present

        """
        return (
            self.pre_rotor_models is not None and len(self.pre_rotor_models.models) > 0
        )

    @property
    def has_post_rotor_models(self) -> bool:
        """
        Flag for having post-rotor models

        Returns
        -------
        has_post_rotor_models
            True if post-rotor models are present

        """
        return (
            self.post_rotor_models is not None
            and len(self.post_rotor_models.models) > 0
        )

    def find_turbine_types(self, algo: Algorithm) -> None:
        """
        Collect the turbine types.

        Parameters
        ----------
        algo
            The algorithm.

        """

        # check turbine models, and find turbine types and pre/post-rotor models:
        turbine_types: list[TurbineType | None] = [None for t in algo.farm.turbines]
        for ti, t in enumerate(algo.farm.turbines):
            for mname in t.models:
                if mname in algo.mbook.turbine_types:
                    m = algo.mbook.turbine_types[mname]
                    if not isinstance(m, TurbineType):
                        raise TypeError(
                            f"Model {mname} type {type(m).__name__} is not derived from {TurbineType.__name__}"
                        )
                    if turbine_types[ti] is not None:
                        prev_tt = turbine_types[ti]
                        assert prev_tt is not None
                        raise TypeError(
                            f"Two turbine type models found for turbine {ti}: {prev_tt.name} and {mname}"
                        )
                    m.name = mname
                    turbine_types[ti] = m

            if turbine_types[ti] is None:
                raise ValueError(
                    f"Turbine {ti}, {t.name}: Missing a turbine type model among models {t.models}"
                )

        self.turbine_types = cast(list[TurbineType], turbine_types)

    def collect_models(self, algo: Algorithm) -> None:
        """
        Analyze and gather turbine models based on the wind farm turbines.

        Parameters
        ----------
        algo
            The calculation algorithm.

        """

        if self.turbine_types is None:
            self.find_turbine_types(algo)

        # check turbine models, and find turbine types and pre/post-rotor models:
        prer_models: list[list[TurbineModel]] = [[] for t in algo.farm.turbines]
        postr_models: list[list[TurbineModel]] = [[] for t in algo.farm.turbines]
        assert self.turbine_types is not None
        ttypes = {m.name: m for m in self.turbine_types}
        rotor_model = algo.rotor_model
        rotor_inputs = set(rotor_model.input_variables())
        for ti, t in enumerate(algo.farm.turbines):
            mlist: list[TurbineModel] = []
            ttp = None
            for mi, mname in enumerate(t.models):
                models: list[TurbineModel]
                if mname in ttypes:
                    models = [ttypes[mname]]
                    ttp = mname
                elif mname in algo.mbook.turbine_models:
                    m = algo.mbook.turbine_models[mname]
                    models_raw = m.models if isinstance(m, FarmDataModelList) else [m]
                    models = [cast(TurbineModel, mm) for mm in models_raw]
                    for mm in models:
                        if not isinstance(mm, TurbineModel):
                            raise TypeError(
                                f"Model {mname} type {type(mm).__name__} is not derived from {TurbineModel.__name__}"
                            )
                    m.name = mname
                else:
                    raise KeyError(
                        f"Model {mname} not found in model book types or models"
                    )
                mlist += models

            # find last model that has rotor inputs,
            # and split pre/post-rotor models there:
            mi = len(mlist) - 1
            while mi >= 0:
                m = mlist[mi]
                ovars = m.output_farm_vars(algo)
                if len(rotor_inputs.intersection(ovars)) > 0:
                    break
                mi -= 1
            prer_models[ti] = mlist[: mi + 1]
            postr_models[ti] = mlist[mi + 1 :]
            assert ttp not in prer_models[ti], (
                f"Turbine type model {ttp} of turbine {ti} cannot be a pre-rotor model. "
                f"Please check turbine model order {[m.name for m in mlist]}, especially "
                f"'{ttp}' and '{prer_models[ti][-1].name}', "
                "and make sure that all models that compute rotor input variables "
                f"{rotor_inputs} appear at the beginning of the model list. "
                f"Identified pre-rotor models: {[m.name for m in prer_models[ti]]}"
            )

        # analyze models:
        mnames_pre, tmsels_pre = self._analyze_models(
            algo, pre_rotor=True, models=prer_models
        )
        mnames_post, tmsels_post = self._analyze_models(
            algo, pre_rotor=False, models=postr_models
        )
        tmsels = tmsels_pre + tmsels_post
        self._tmall = [bool(np.all(t)) for t in tmsels]
        self.turbine_model_names = mnames_pre + mnames_post
        if len(self.turbine_model_names):
            self._tmsels = {mi: t for mi, t in enumerate(tmsels) if not self._tmall[mi]}
        else:
            raise ValueError(f"Controller '{self.name}': No turbine model found.")

    def __get_pars(
        self,
        algo: Algorithm,
        models: list[Any],
        ptype: str,
        mdata: MData | None = None,
        downwind_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Private helper function for gathering model parameters.
        """
        pars = []
        assert self.turbine_model_names is not None
        assert self._tmall is not None
        for m in models:
            mi = self.turbine_model_names.index(m.name)
            if self._tmall[mi]:
                s = np.s_[:, :] if downwind_index is None else np.s_[:, downwind_index]
            else:
                assert mdata is not None
                vsel = self._tmodel_sels_var(mi)
                if downwind_index is None:
                    s = mdata[vsel]
                else:
                    s = np.s_[mdata[vsel][:, downwind_index], downwind_index]
            pars.append({"st_sel": s})
            if m.name in self.pars:
                pars[-1].update(self.pars[m.name][ptype])

        return pars

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initialize the model.

        Parameters
        ----------
        algo
            The calculation algorithm.
        verbosity
            The verbosity level, where 0 is silent.
        loaded_data
            Data that has already been loaded, to be extended by this function.
            It contains coordinate data, model variables, and additional data.
        force
            Overwrite existing data.

        Returns
        -------
        loaded_data
            The loaded data, containing the keys "coords", "data_vars", and
            "extra_data".

        """
        self.collect_models(algo)
        return super().initialize(
            algo=algo,
            loaded_data=loaded_data,
            force=force,
            verbosity=verbosity,
        )

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 0,
    ) -> None:
        """
        Load and/or create the model data subject to chunking.

        Such data should not be stored under self for memory reasons. The data
        returned here will automatically be chunked and then provided as part of
        the mdata object during calculations.

        Parameters
        ----------
        algo
            The calculation algorithm.
        loaded_data
            Data that has already been loaded, to be extended by this function.
            It contains coordinate data, model variables, and additional data.
        force
            Overwrite existing data.
        verbosity
            The verbosity level, where 0 is silent.

        """
        if force or FC.TMODELS not in loaded_data["coords"]:
            super().load_data(algo, loaded_data, force=force, verbosity=verbosity)

            assert self.turbine_model_names is not None
            loaded_data["coords"][FC.TMODELS] = np.asarray(
                self.turbine_model_names, dtype=str
            )
            for mi, tsel in (self._tmsels or {}).items():
                loaded_data["data_vars"][self._tmodel_sels_var(mi)] = (
                    (FC.STATE, FC.TURBINE),
                    tsel,
                )
            loaded_data["data_vars"].pop(FC.TMODEL_SELS, None)
            self._tmsels = None

    def output_farm_vars(self, algo: Algorithm) -> list[str]:
        """
        Return the variables modified by the model.

        Parameters
        ----------
        algo
            The calculation algorithm.

        Returns
        -------
        output_vars
            The output variable names.

        """
        assert self.pre_rotor_models is not None
        assert self.post_rotor_models is not None
        ovars = set(self.pre_rotor_models.output_farm_vars(algo))
        ovars.update(self.post_rotor_models.output_farm_vars(algo))

        return list(ovars)

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        pre_rotor: bool,
        downwind_index: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Run the main model calculation for a single chunk of data.

        This function is executed on a single chunk of data, and all
        computations should be based on numpy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            The model data.
        fdata
            The farm data.
        pre_rotor
            Flag indicating whether to run pre-rotor or post-rotor models.
        downwind_index
            The index in the downwind order.

        Returns
        -------
        results
            The resulting data. Keys are output variable names and values are
            arrays with shape ``(n_states, n_turbines)``.

        """
        s = self.pre_rotor_models if pre_rotor else self.post_rotor_models
        assert s is not None
        pars = self.__get_pars(algo, s.models, "calc", mdata, downwind_index)
        res = s.calculate(algo, mdata, fdata, parameters=pars)
        return res

    def finalize(self, algo: Algorithm, verbosity: int = 0) -> None:
        """
        Finalize the model.

        Parameters
        ----------
        algo
            The calculation algorithm.
        verbosity
            The verbosity level, where 0 is silent.

        """
        super().finalize(algo, verbosity)
        self.turbine_model_names = None

    @classmethod
    def new(
        cls,
        controller_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> FarmController:
        """
        Create a farm controller instance at runtime.

        Parameters
        ----------
        controller_type
            The selected derived class name.
        args
            Additional positional arguments for the constructor.
        kwargs
            Additional keyword arguments for the constructor.

        """
        return cast(FarmController, new_instance(cls, controller_type, *args, **kwargs))
