import numpy as np

from .self_similar import SelfSimilar


class SelfSimilar2020(SelfSimilar):
    """
    The self-similar 2020 induction wake model
    from Troldborg and Meyer Forsting

    The individual wake effects are superposed linearly,
    without invoking a wake superposition model.

    Notes
    -----
    References:
    [1] Troldborg, Niels, and Alexander Raul Meyer Forsting.
    "A simple model of the wind turbine induction zone derived from numerical simulations."
    Wind Energy 20.12 (2017): 2011-2020.
    https://onlinelibrary.wiley.com/doi/full/10.1002/we.2137

    [2] Forsting, Alexander R. Meyer, et al.
    "On the accuracy of predicting wind-farm blockage."
    Renewable Energy (2023).
    https://www.sciencedirect.com/science/article/pii/S0960148123007620

    :group: models.wake_models.induction

    """

    def _a0(self, ct, x_R, gamma=1.1):
        """Helper function: define a0 with gamma factor, eqn 8 from [2]"""

        x_new = np.minimum(np.maximum(-1 * np.abs(x_R), -6), -1)
        c = (self._mu(x_new) - self._mu(-1)) / (self._mu(-6) - self._mu(-1))

        fg1 = -0.06489
        fg2 = 0.4911
        fg3 = 0.1577
        fg4 = 1.116
        far_gamma = fg1 * np.sin((ct - fg2) / fg3) + fg4

        ng1 = -1.381
        ng2 = 2.627
        ng3 = -1.524
        ng4 = 1.336
        near_gamma = ng1 * ct**3 + ng2 * ct**2 + (ng3 * ct) + ng4

        gamma = c * far_gamma + (1 - c) * near_gamma

        return self.induction.ct2a(gamma * ct)

    def _r_half(self, x_R):
        """Helper function: define induction zone half radius (eqn 13)"""
        return -0.672 * x_R + 0.4897
