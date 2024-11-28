from matplotlib import pyplot

from .output import Output


class plt(Output):
    """
    Class that runs plt commands

    :group: output

    """

    def __getattr__(self, name):
        return getattr(pyplot, name)

    def savefig(self, fname, *args, **kwargs):
        fpath = super().get_fpath(fname)
        pyplot.savefig(fpath, *args, **kwargs)
