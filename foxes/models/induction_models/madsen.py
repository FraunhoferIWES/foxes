from foxes.core import InductionModel

class MadsenInduction(InductionModel):
    """
    Computes the induction factor through polynomial
    fit, extending validity for high ct values

    Attributes
    ----------
    k1: float
        Model coefficient
    k2: float
        Model coefficient
    k3: float
        Model coefficient

    :group: models.induction

    """
    def __init__(self, k1=0.2460, k2=0.0586, k3=0.0883):
        """
        Constructor.
        
        Parameters
        ----------
        k1: float
            Model coefficient
        k2: float
            Model coefficient
        k3: float
            Model coefficient

        """
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def ct2a(self, ct):
        """
        Computes induction from ct

        Parameters
        ----------
        ct: numpy.ndarray or float
            The ct values
        
        Returns
        -------
        ct: numpy.ndarray or float
            The induction values

        """
        return self.k1*ct + self.k2*ct**2 + self.k3*ct**3
