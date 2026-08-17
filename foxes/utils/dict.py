from __future__ import annotations

from yaml import safe_load
from pathlib import Path
from typing import Any, TypeVar, cast

from .exec_python import eval_dict_values


K = TypeVar("K")
V = TypeVar("V")


class Dict(dict[K, V]):
    """
    A slightly enhanced dictionary.

    :group: utils

    """

    def __init__(self, *args: Any, _name: str | None = None, **kwargs: Any) -> None:
        """
        Constructor.

        Parameters
        ----------
        *args: tuple, optional
            Arguments passed to `dict`
        _name
            The dictionary name
        **kwargs: dict, optional
            Arguments passed to `dict`

        """
        super().__init__()
        self._name = _name if _name is not None else type(self).__name__
        self.update(*args, **kwargs)

    @property
    def name(self) -> str:
        """
        The dictionary name

        Returns
        -------
        name
            The dictionary name

        """
        return self._name

    def get_item(self, key: Any, *deflt: Any, prnt: bool = True) -> Any:
        """
        Gets an item, prints readable error if not found

        Parameters
        ----------
        key: immutable object
            The key
        deflt
            A single-item sequence containing the default
        prnt
            Flag for message printing

        Returns
        -------
        data
            The data

        """
        try:
            if len(deflt):
                assert len(deflt) == 1, (
                    f"Expecting a single default entry, got {len(deflt)}"
                )
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
            data = Dict(data, _name=f"{self.name}.{key}")

        return data

    def pop_item(self, key: Any, *deflt: Any, prnt: bool = True) -> Any:
        """
        Pops an item, prints readable error if not found

        Parameters
        ----------
        key: immutable object
            The key
        deflt
            A single-item sequence containing the default
        prnt
            Flag for message printing

        Returns
        -------
        data
            The data

        """
        data = self.get_item(key, *deflt, prnt=prnt)
        if key in self:
            del self[key]
        return data

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, list):
            out_list: list[Any] = []
            for i, x in enumerate(value):
                if isinstance(x, dict) and not isinstance(x, Dict):
                    nme = f"{self.name}.{key}"
                    if len(value) > 1:
                        nme += f".{i}"
                    out_list.append(Dict(x, _name=nme))
                else:
                    out_list.append(x)
            value = out_list
        elif isinstance(value, dict) and not isinstance(value, Dict):
            out_dict: Dict[Any, Any] = Dict(_name=f"{self.name}.{key}")
            out_dict.update(value)
            value = out_dict

        super().__setitem__(key, value)

    def __getitem__(self, key: Any) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            k = ", ".join(sorted([f"{s}" for s in self.keys()]))
            e = f"{self.name}: Cannot find key '{key}'. Known keys: {k}"
            raise KeyError(e)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Update the dictionary with the key/value pairs from other, overwriting existing keys.
        """
        other = dict(*args, **kwargs)
        for k, v in other.items():
            self[k] = v

    def eval(
        self,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
    ) -> Dict[Any, Any]:
        """
        Tries to evaluate all string values, recursively.

        Parameters
        ----------
        globals
            The global namespace
        locals
            The local namespace

        Returns
        -------
        self
            The dictionary with evaluated values

        """
        return Dict(
            eval_dict_values(cast(dict[str, Any], self), globals, locals),
            _name=self.name,
        )

    @classmethod
    def from_yaml(self, yml_file: str | Path, verbosity: int = 1) -> Dict[Any, Any]:
        """
        Reads a yaml file

        Parameters
        ----------
        yml_file
            Path to the yaml file
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        dct
            The data

        """

        def _print(*args: Any, level: int = 1, **kwargs: Any) -> None:
            if verbosity >= level:
                print(*args, **kwargs)

        fpath = Path(yml_file)
        _print("Reading file", fpath)
        with open(fpath) as stream:
            data = safe_load(stream)
        if data is None:
            data = {}
        dct: Dict[Any, Any] = Dict(data, _name=fpath.stem)
        _print(dct, level=2)

        return dct
