class Dict(dict):
    """
    A slightly enhanced dictionary.

    Attributes
    ----------
    name: str
        The dictionary name

    :group: utils

    """

    def __init__(self, *args, name=None, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        *args: tuple, optional
            Arguments passed to `dict`
        name: str, optional
            The dictionary name
        **kwargs: dict, optional
            Arguments passed to `dict`

        """
        super().__init__(*args, **kwargs)
        self.name = name if name is not None else type(self).__name__

    def __getitem__(self, key, prnt=True):
        try:
            return super().__getitem__(key)
        except KeyError:

            if prnt:
                print(f"\n{self.name}: Cannot find key '{key}'.\n")
                print("Known keys:")
                for k in self.keys():
                    print("   ", k)
                print()

            k = ", ".join(sorted([f"{s}" for s in self.keys()]))
            e = f"{self.name}: Cannot find key '{key}'. Known keys: {k}"
            raise KeyError(e)
