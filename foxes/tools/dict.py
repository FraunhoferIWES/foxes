class Dict(dict):
    """
    A slightly enhanced dictionary.

    Parameters
    ----------
    *args : tuple, optional
        Arguments passed to `dict`
    name : str, optional
        The dictionary name
    **kwargs : dict, optional
        Arguments passed to `dict`

    Attributes
    ----------
    name : str
        The dictionary name

    """

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name if name is not None else type(self).__name__

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            k = ", ".join(sorted(list(self.keys())))
            e = f"{self.name}: Cannot find key '{key}'. Known keys: {k}"
            raise KeyError(e)
