import numpy as np
from xarray import Dataset
from io import StringIO
from pathlib import Path


def read_tab_file(fname: str | Path, normalize: bool = True) -> Dataset:
    """
    Reads a tab file into a Dataset

    Parameters
    ----------
    fname
        The path to the tab file
    normalize
        Normalize the frequencies such that
        they add to 1000 in each sector

    Returns
    -------
    out
        The data


    """

    header = []
    with open(fname, "r") as f:
        header.append(f.readline().replace("\t", " ").strip())
        header.append(f.readline().replace("\t", " ").strip())
        header.append(f.readline().replace("\t", " ").strip())
        sfreqs_text = f.readline().replace("\t", " ").strip()
        s = f.read().replace("\t", " ").strip()
    data = np.genfromtxt(StringIO(s))
    sfreqs = np.fromstring(sfreqs_text, sep=" ")

    descr = header[0]
    lat, lon, height = np.fromstring(header[1], sep=" ")
    nsec, a, b = np.fromstring(header[2], sep=" ")

    delta_wd = 360 / nsec
    out = Dataset(
        coords={"ws": data[:, 0], "wd": np.arange(0, 360.0, delta_wd)},
        data_vars={
            "wd_freq": (("wd",), sfreqs),
            "ws_freq": (("ws", "wd"), data[:, 1:]),
        },
        attrs={
            "description": descr,
            "latitude": lat,
            "longitude": lon,
            "height": height,
            "factor_ws": a,
            "shift_wd": b,
        },
    )

    if normalize:
        out["ws_freq"] *= 1000 / np.sum(data[:, 1:], axis=0)[None, :]
        out["wd_freq"] *= 100 / np.sum(sfreqs)

    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fname", help="Path to the tab file")
    args = parser.parse_args()

    read_tab_file(args.fname)
