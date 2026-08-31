from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any

from foxes.core import WakeK
from foxes.models.wake_models.gaussian import GaussianWakeModel
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData, Model
    from foxes.core.wake_model import WakeModel


class TurbOParkWake(GaussianWakeModel):
    """
    The TurbOPark wake model

    Notes
    -----
    Reference:
    "Turbulence Optimized Park model with Gaussian wake profile"
    J G Pedersen, E Svensson, L Poulsen and N G Nygaard
    https://iopscience.iop.org/article/10.1088/1742-6596/2265/2/022063/pdf

    Attributes
    ----------
    sbeta_factor
        Factor multiplying sbeta
    c1
        Factor from Frandsen turbulence model
    c2
        Factor from Frandsen turbulence model
    induction
        The induction model
    wake_k
        Handler for the wake growth parameter k


    """

    def __init__(
        self,
        superposition: str,
        sbeta_factor: float = 0.25,
        c1: float = 1.5,
        c2: float = 0.8,
        induction: str = "Madsen",
        **wake_k: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        superposition
            The wind deficit superposition
        sbeta_factor
            Factor multiplying sbeta
        c1
            Factor from Frandsen turbulence model
        c2
            Factor from Frandsen turbulence model
        induction
            The induction model
        wake_k
            Parameters for the WakeK class

        """
        super().__init__(wind_superposition=superposition)

        self.sbeta_factor = sbeta_factor
        self.c1 = c1
        self.c2 = c2
        self.induction = induction
        self.wake_k = WakeK(**wake_k)

    def __repr__(self) -> str:
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        s = f"{type(self).__name__}"
        s += f"({self.wind_superposition}, induction={iname}, "
        s += self.wake_k.repr() + ")"
        return s

    def waked_variables(self) -> list[str]:
        """
        Returns a list of variable names that are affected by this wake model.

        Returns
        -------
        waked_variables
            A list of variable names affected by this wake model.

        """
        return [FV.WS]

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            All sub models

        """
        smdls: list[Model] = [self.wake_k]
        if not isinstance(self.induction, str):
            smdls.append(self.induction)
        return smdls

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        if isinstance(self.induction, str):
            self.induction = algo.mbook.axial_induction[self.induction]
        return super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

    def calc_amplitude_sigma(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        x: np.ndarray,
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """
        Calculate the amplitude and the sigma,
        both depend only on x (not on r).

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        downwind_index
            The index in the downwind order
        x
            The x values, shape: (n_states, n_targets)

        Returns
        -------
        amsi
            The amplitude and sigma arrays
            with shape (n_st_sel,)
        st_sel
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        """
        assert not isinstance(self.induction, str)

        # get ct:
        ct = self.get_data(
            FV.CT,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )

        # select targets:
        st_sel = (x > 1e-8) & (ct > 1e-8)
        if np.any(st_sel):
            # apply selection:
            x = x[st_sel]
            ct = ct[st_sel]

            # get D:
            D = self.get_data(
                FV.D,
                FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=False,
                selection=st_sel,
            )

            # get TI:
            ati = self.get_data(
                FV.AMB_TI,
                FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=True,
            )

            # get k:
            k = self.wake_k(
                FC.STATE_TARGET,
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                amb_ti=ati,
                upcast=False,
                selection=st_sel,
            )

            # calculate sigma:
            # beta = np.sqrt(0.5 * (1 + np.sqrt(1.0 - ct)) / np.sqrt(1.0 - ct))
            a = self.induction.ct2a(ct)
            beta = np.maximum((1 - a) / (1 - 2 * a), 0)
            epsilon = self.sbeta_factor * np.sqrt(beta)
            del a, beta

            ati = ati[st_sel]
            alpha = self.c1 * ati
            beta = self.c2 * ati / np.sqrt(ct)

            # calculate sigma (eqn 4)
            sigma = D * (
                epsilon
                + k
                / beta
                * (
                    np.sqrt((alpha + beta * x / D) ** 2 + 1)
                    - np.sqrt(1 + alpha**2)
                    - np.log(
                        (np.sqrt((alpha + beta * x / D) ** 2 + 1) + 1)
                        * alpha
                        / ((np.sqrt(1 + alpha**2) + 1) * (alpha + beta * x / D))
                    )
                )
            )

            del (
                x,
                alpha,
                beta,
                epsilon,
            )

            # calculate amplitude, same as in Bastankhah model (eqn 7)
            ct_eff = ct / (8 * (sigma / D) ** 2)
            ampld = np.maximum(-2 * self.induction.ct2a(ct_eff), -1)

        # case no targets:
        else:
            st_sel = np.zeros_like(x, dtype=bool)
            n_sp = np.sum(st_sel)
            ampld = np.zeros(n_sp, dtype=config.dtype_double)
            sigma = np.zeros(n_sp, dtype=config.dtype_double)

        return {FV.WS: (ampld, sigma)}, st_sel


class TurbOParkWakeIX(GaussianWakeModel):
    """
    The generalized TurbOPark wake model, integrating TI over the streamline.

    https://iopscience.iop.org/article/10.1088/1742-6596/2265/2/022063/pdf

    Attributes
    ----------
    dx
        The step size of the integral
    sbeta_factor
        Factor multiplying sbeta
    self_wake
        Flag for considering only own wake in ti integral
    induction
        The induction model
    ipars
        Additional parameters for centreline integration
    wake_k
        Handler for the wake growth parameter k


    """

    def __init__(
        self,
        superposition: str,
        dx: float,
        sbeta_factor: float = 0.25,
        self_wake: bool = True,
        induction: str = "Madsen",
        ipars: dict[str, Any] | None = None,
        **wake_k: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        superposition
            The wind deficit superposition
        dx
            The step size of the integral
        sbeta_factor
            Factor multiplying sbeta
        self_wake
            Flag for considering only own wake in ti integral
        induction
            The induction model
        ipars
            Additional parameters for centreline integration
        wake_k
            Parameters for the WakeK class

        """
        super().__init__(wind_superposition=superposition)

        self.dx = dx
        self.sbeta_factor = sbeta_factor
        self.ipars = {} if ipars is None else ipars
        self._tiwakes: list[WakeModel] | None = None
        self.self_wake = self_wake
        self.induction = induction
        self.wake_k = WakeK(**wake_k)

        assert not self.wake_k.is_kTI, f"{self.name}: Cannot apply ka or ambka setup"

    def __repr__(self) -> str:
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        s = f"{type(self).__name__}"
        s += f"({self.wind_superposition}, induction={iname}, dx={self.dx}, "
        s += self.wake_k.repr() + ")"
        return s

    def waked_variables(self) -> list[str]:
        """
        Returns a list of variable names that are affected by this wake model.

        Returns
        -------
        waked_variables
            A list of variable names affected by this wake model.

        """
        return [FV.WS]

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            All sub models

        """
        smdls: list[Model] = [self.wake_k]
        if not isinstance(self.induction, str):
            smdls.append(self.induction)
        return smdls

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        if isinstance(self.induction, str):
            self.induction = algo.mbook.axial_induction[self.induction]
        return super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

    def new_wake_deltas(
        self, algo: Algorithm, mdata: MData, fdata: FData, tdata: TData
    ) -> dict[str, np.ndarray]:
        """
        Creates new empty wake delta arrays.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data

        Returns
        -------
        wake_deltas
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_turbines, n_rpoints, ...)

        """
        # find TI wake model:
        self._tiwakes = []
        for w in algo.wake_models.values():
            if w is not self:
                wdel = w.new_wake_deltas(algo, mdata, fdata, tdata)
                if self.wake_k.ti_var in wdel:
                    self._tiwakes.append(w)
        if self.wake_k.ti_var not in FV.amb2var and len(self._tiwakes) == 0:
            raise KeyError(
                f"Model '{self.name}': Missing wake model that computes wake delta for variable {self.wake_k.ti_var}"
            )

        return super().new_wake_deltas(algo, mdata, fdata, tdata)

    def calc_amplitude_sigma(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        x: np.ndarray,
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """
        Calculate the amplitude and the sigma,
        both depend only on x (not on r).

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        downwind_index
            The index in the downwind order
        x
            The x values, shape: (n_states, n_targets)

        Returns
        -------
        amsi
            The amplitude and sigma arrays
            with shape (n_st_sel,)
        st_sel
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        """
        assert not isinstance(self.induction, str)

        # get ct:
        ct = self.get_data(
            FV.CT,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )

        # select targets:
        st_sel = (x > 1e-8) & (ct > 1e-8)
        if np.any(st_sel):
            # apply selection:
            # x = x[st_sel]
            ct = ct[st_sel]

            # get D:
            D = self.get_data(
                FV.D,
                FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=False,
                selection=st_sel,
            )

            # get k:
            k = self.wake_k(
                FC.STATE_TARGET,
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=False,
                selection=st_sel,
            )

            # calculate sigma:
            # beta = np.sqrt(0.5 * (1 + np.sqrt(1.0 - ct)) / np.sqrt(1.0 - ct))
            a = self.induction.ct2a(ct)
            beta = (1 - a) / (1 - 2 * a)
            epsilon = self.sbeta_factor * np.sqrt(beta)
            del a, beta

            # get TI by integration along centre line:
            ti_ix = algo.wake_frame.calc_centreline_integral(
                algo,
                mdata,
                fdata,
                downwind_index,
                [self.wake_k.ti_var],
                x,
                dx=self.dx,
                wake_models=self._tiwakes,
                self_wake=self.self_wake,
                **self.ipars,
            )[:, :, 0]

            # calculate sigma (eqn 1, plus epsilon from eqn 4 for x = 0)
            sigma = D * epsilon + k * ti_ix[st_sel]
            del x, epsilon

            # calculate amplitude, same as in Bastankhah model (eqn 7)
            ct_eff = ct / (8 * (sigma / D) ** 2)
            ampld = np.maximum(-2 * self.induction.ct2a(ct_eff), -1)

        # case no targets:
        else:
            st_sel = np.zeros_like(x, dtype=bool)
            n_sp = np.sum(st_sel)
            ampld = np.zeros(n_sp, dtype=config.dtype_double)
            sigma = np.zeros(n_sp, dtype=config.dtype_double)

        return {FV.WS: (ampld, sigma)}, st_sel

    def finalize(self, algo: Algorithm, verbosity: int = 0) -> None:
        """
        Finalizes the model.

        Parameters
        ----------
        algo
            The calculation algorithm
        verbosity
            The verbosity level, 0 = silent

        """
        super().finalize(algo, verbosity)
        self._tiwakes = None
