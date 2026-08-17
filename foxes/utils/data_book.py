from __future__ import annotations

import importlib.resources as resources

from pathlib import Path
from typing import Any


class DataBook:
    """
    Container class for file paths, either directly
    given or as static data within a package.

    Parameters
    ----------
    data_book
        A data book to start from

    Attributes
    ----------
    dbase
        The data base. Key: context str,
        value

    :group: utils

    """

    def __init__(self, data_book: DataBook | None = None) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_book: DataBook, optional
            A data book to start from

        """
        self.dbase: dict[str, dict[str, Path]] = {}
        if data_book is not None:
            for c, d in data_book.dbase.items():
                self.dbase[c] = {}
                self.dbase[c].update(d)

    def add_data_package(
        self, context: str, package: str | Any, file_sfx: str | list[str]
    ) -> None:
        """
        Add static files from a package location.

        Parameters
        ----------
        context
            The context
        package
            The package, must contain init file
        file_sfx
            File endings to include

        """
        if context not in self.dbase:
            self.dbase[context] = {}

        if isinstance(file_sfx, str):
            file_sfx = [file_sfx]

        try:
            contents = [
                r.name for r in resources.files(package).iterdir() if r.is_file()
            ]
        except AttributeError:
            contents = list(resources.contents(package))

        def check_f(f: str) -> bool:
            """little helper function to check file endings"""
            return any([len(f) > len(s) and f[-len(s) :] == s for s in file_sfx])

        contents = [f for f in contents if check_f(f)]

        try:
            for f in contents:
                with resources.as_file(resources.files(package).joinpath(f)) as path:
                    self.dbase[context][f] = path
        except AttributeError:
            for f in contents:
                with resources.path(package, f) as path:
                    self.dbase[context][f] = path

    def add_data_package_file(
        self, context: str, package: str | Any, file_name: str
    ) -> None:
        """
        Add a static file from a package location.

        Parameters
        ----------
        context
            The context
        package
            The package, must contain init.py file
        file_mane
            The file name

        """
        if context not in self.dbase:
            self.dbase[context] = {}

        try:
            with resources.path(package, file_name) as path:
                self.dbase[context][file_name] = path
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File '{file_name}' not found in package '{package}'"
            )

    def add_files(self, context: str, file_paths: list[str]) -> None:
        """
        Add file paths

        Parameters
        ----------
        context
            The context
        file_paths
            The file paths

        """

        if context not in self.dbase:
            self.dbase[context] = {}

        for f in file_paths:
            path = Path(f)
            if not path.is_file():
                raise FileNotFoundError(
                    f"File '{path}' not found, cannot add to context '{context}'"
                )
            self.dbase[context][path.name] = path

    def add_file(self, context: str, file_path: str) -> None:
        """
        Add a file path

        Parameters
        ----------
        context
            The context
        file_path
            The file path

        """
        self.add_files(context, [file_path])

    def get_file_path(
        self,
        context: str,
        file_name: str,
        check_raw: bool = True,
        errors: bool = True,
    ) -> Path | None:
        """
        Get path of a file

        Parameters
        ----------
        context
            The context
        file_name
            The file name
        check_raw
            Check if `file_name` exists as given, and in
            that case return the path
        errors
            Flag for raising KeyError, otherwise return None,
            if context of file_name not found

        Returns
        -------
        path
            The path

        """

        if check_raw:
            path = Path(file_name)
            if path.is_file():
                return path

        file_name = str(file_name)

        try:
            cdata = self.dbase[context]
        except KeyError:
            if not errors:
                return None
            raise KeyError(
                f"Context '{context}' not found in data book. Available: {sorted(list(self.dbase.keys()))}"
            )

        try:
            return cdata[file_name]
        except KeyError:
            if not errors:
                return None
            raise KeyError(
                f"File '{file_name}' not found in context '{context}'. Available: {sorted(list(cdata.keys()))}"
            )

    def toc(self, context: str) -> list[str]:
        """
        Get list of contents

        Parameters
        ----------
        context
            The context

        Returns
        -------
        keys
            The data keys

        """
        return sorted(list(self.dbase[context].keys()))
