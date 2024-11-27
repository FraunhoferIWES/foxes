from matplotlib import pyplot

from .output import Output


class plt(Output):
    """
    Class that runs plt commands

    :group: output

    """

    def figure(self, *args, **kwargs):
        """Runs pyplot.figure"""
        return pyplot.figure(*args, **kwargs)
    
    def subplot(self, *args, **kwargs):
        """Runs pyplot.subplot"""
        return pyplot.subplot(*args, **kwargs)
    
    def subplots(self, *args, **kwargs):
        """Runs pyplot.subplots"""
        return pyplot.subplots(*args, **kwargs)
    
    def show(self, *args, **kwargs):
        """Runs pyplot.show"""
        pyplot.show(*args, **kwargs)

    def close(self, *args, **kwargs):
        """Runs pyplot.close"""
        pyplot.close(*args, **kwargs)
