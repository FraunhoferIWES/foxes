from yaml import safe_load
from pathlib import Path


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
        tmp = dict()
        for a in args:
            tmp.update(
                {
                    k: (
                        Dict(d, name=f"{name}.{k}")
                        if isinstance(d, dict) and not isinstance(d, Dict)
                        else d
                    )
                    for k, d in a.items()
                }
            )

        super().__init__(
            **tmp,
            **{
                k: (
                    Dict(d, name=k)
                    if isinstance(d, dict) and not isinstance(d, Dict)
                    else d
                )
                for k, d in kwargs.items()
            },
        )

        self.name = name if name is not None else type(self).__name__

    def get_item(self, key, *deflt, prnt=True):
        """
        Gets an item, prints readable error if not found

        Parameters
        ----------
        key: immutable object
            The key
        deflt: tuple, optional
            Tuple of length 1, containing the default
        prnt: bool
            Flag for message printing

        Returns
        -------
        data: object
            The data

        """
        try:
            if len(deflt):
                assert (
                    len(deflt) == 1
                ), f"Expecting a single default entry, got {len(deflt)}"
                data = self.get(key, deflt[0])
            else:
                data = self[key]
        except KeyError as e:
            if prnt:
                print(f"\n{self.name}: Cannot find key '{key}'.\n")
                print("Known keys:")
                for k in self.keys():
                    print("   ", k)
                print()
            raise e

        if isinstance(data, dict) and not isinstance(data, Dict):
            data = Dict(data, name=f"{self.name}.{key}")

        return data

    def pop_item(self, key, *deflt, prnt=True):
        """
        Pops an item, prints readable error if not found

        Parameters
        ----------
        key: immutable object
            The key
        deflt: tuple, optional
            Tuple of length 1, containing the default
        prnt: bool
            Flag for message printing

        Returns
        -------
        data: object
            The data

        """
        data = self.get_item(key, *deflt, prnt=prnt)
        if key in self:
            del self[key]
        return data

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            k = ", ".join(sorted([f"{s}" for s in self.keys()]))
            e = f"{self.name}: Cannot find key '{key}'. Known keys: {k}"
            raise KeyError(e)

    @classmethod
    def from_yaml(self, yml_file, verbosity=1):
        """
        Reads a yaml file

        Parameters
        ----------
        yml_file: str
            Path to the yaml file
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        dct: Dict
            The data

        """

        def _print(*args, level=1, **kwargs):
            if verbosity >= level:
                print(*args, **kwargs)

        fpath = Path(yml_file)
        _print("Reading file", fpath)
        with open(fpath) as stream:
            data = safe_load(stream)
        if data is None:
            data = {}
        dct = Dict(data, name=fpath.stem)
        _print(dct, level=2)

        return dct
