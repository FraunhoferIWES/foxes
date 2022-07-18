import numpy as np

from foxes.core import VerticalProfile


class UniformProfile(VerticalProfile):
    def __init__(self, variable):
        super().__init__(self)
        self.var = variable

    def input_vars(self):
        return [self.var]

    def calculate(self, data, heights):
        out = np.zeros_like(heights)
        out[:] = data[self.var]
        return out
