import importlib.resources as resources

from pathlib import Path


class DataBook:
    """
    Container class for file paths, either directly
    given or as static data within a package.

    Parameters
    ----------
    data_book: DataBook, optional
        A data book to start from

    Attributes
    ----------
    dbase: dict
        The data base. Key: context str,
        value: dict (file name str to pathlib.Path)

    :group: utils

    """

    def __init__(self, data_book=None):
        """
        Constructor.

        Parameters
        ----------
        data_book: DataBook, optional
            A data book to start from

        """
        self.dbase = {}
        if data_book is not None:
            for c, d in data_book.items():
                self.dbase[c] = {}
                self.dbase[c].update(d)

    def add_data_package(self, context, package, file_sfx):
        """
        Add static files from a package location.

        Parameters
        ----------
        context: str
            The context
        package: str or package
            The package, must contain init file
        file_sfx: list of str
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

        check_f = lambda f: any(
            [len(f) > len(s) and f[-len(s) :] == s for s in file_sfx]
        )
        contents = [f for f in contents if check_f(f)]

        try:
            for f in contents:
                with resources.as_file(resources.files(package).joinpath(f)) as path:
                    self.dbase[context][f] = path
        except AttributeError:
            for f in contents:
                with resources.path(package, f) as path:
                    self.dbase[context][f] = path

    def add_data_package_file(self, context, package, file_name):
        """
        Add a static file from a package location.

        Parameters
        ----------
        context: str
            The context
        package: str or package
            The package, must contain init.py` file
        file_mane: str
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

    def add_files(self, context, file_paths):
        """
        Add file paths

        Parameters
        ----------
        context: str
            The context
        file_paths: list of str
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

    def add_file(self, context, file_path):
        """
        Add a file path

        Parameters
        ----------
        context: str
            The context
        file_path: str
            The file path

        """
        self.add_files(context, [file_path])

    def get_file_path(self, context, file_name, check_raw=True, errors=True):
        """
        Get path of a file

        Parameters
        ----------
        context: str
            The context
        file_name: str
            The file name
        check_raw: bool
            Check if `file_name` exists as given, and in
            that case return the path
        errors: bool
            Flag for raising KeyError, otherwise return None,
            if context of file_name not found

        Returns
        -------
        path: pathlib.Path
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

    def toc(self, context):
        """
        Get list of contents

        Parameters
        ----------
        context: str
            The context

        Returns
        -------
        keys: list of str
            The data keys

        """
        return sorted(list(self.dbase[context].keys()))
