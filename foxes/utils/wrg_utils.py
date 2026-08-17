import pandas as pd
from pathlib import Path


class ReaderWRG:
    """
    A reader for WRG files

    Attributes
    ----------
    fpath
        Path to the wrg file

    """

    def __init__(self, fpath: str | Path) -> None:
        """
        Constructor

        Parameters
        ----------
        fpath
            Path to the wrg file
        """
        self.fpath = Path(fpath)
        self._prepare()

    def _prepare(self) -> None:
        # read the first two lines
        with open(self.fpath, "r") as fstream:
            first_line = fstream.readline().split()
            nx_s, ny_s, utmx0_s, utmy0_s, res_s = first_line
            second_line = fstream.readline().split()
            n_cols = len(second_line)
            self._n_sectors = int(second_line[8])
            self._nx = int(nx_s)
            self._ny = int(ny_s)
            self._utmx0 = float(utmx0_s)
            self._utmy0 = float(utmy0_s)
            self._res = int(res_s)

        def cols_sel(name: str, secs: int) -> list[str]:
            """little helper to create column names"""
            return [f"{name}_{i}" for i in range(secs)]

        cols: list[str] = [""] * (8 + 3 * self._n_sectors)
        cols[0] = "utmx"
        cols[1] = "utmy"
        cols[2] = "z"
        cols[3] = "h"
        cols[4] = "A"
        cols[5] = "K"
        cols[6] = "pw"
        cols[7] = "n_sectors"
        cols[8::3] = cols_sel("fs", self._n_sectors)
        cols[9::3] = cols_sel("As", self._n_sectors)
        cols[10::3] = cols_sel("Ks", self._n_sectors)

        self._data = pd.read_csv(
            self.fpath, names=cols, skiprows=1, sep=r"\s+", usecols=range(1, n_cols)
        )

        self._data[cols_sel("fs", self._n_sectors)] /= 10
        self._data[cols_sel("As", self._n_sectors)] /= 10
        self._data[cols_sel("Ks", self._n_sectors)] /= 100

        if len(self._data.index) != self._nx * self._ny:
            raise ValueError(
                f"Expecting {self._nx * self._ny} rows in data, got {len(self._data.index)}"
            )

    @property
    def data(self) -> pd.DataFrame:
        """
        The WRG data

        Returns
        -------
        df
            The WRG data

        """
        return self._data

    @property
    def nx(self) -> int:
        """
        The number of points in x direction

        Returns
        -------
        n
            The number of points in x direction

        """
        return self._nx

    @property
    def ny(self) -> int:
        """
        The number of points in y direction

        Returns
        -------
        n
            The number of points in y direction

        """
        return self._ny

    @property
    def x0(self) -> float:
        """
        The lower left x coordinate

        Returns
        -------
        x
            The lower left x coordinate

        """
        return self._utmx0

    @property
    def y0(self) -> float:
        """
        The lower left y coordinate

        Returns
        -------
        y
            The lower left y coordinate

        """
        return self._utmy0

    @property
    def n_sectors(self) -> int:
        """
        The number of wind direction sectors

        Returns
        -------
        n
            The number of wind direction sectors

        """
        return self._n_sectors

    @property
    def resolution(self) -> int:
        """
        The horizontal resolution

        Returns
        -------
        res
            The horizontal resolution

        """
        return self._res
