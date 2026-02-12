from pathlib import Path

from .load import import_module


def download_file(url, out_path):
    """
    Download a file from a URL with resume capability.

    Parameters
    ----------
    url: str
        The URL to download from
    out_path: str
        Path to the output file

    Returns
    -------
    scs: int
        Success indicator. 0 = File already there,
        1 = Success, -1 = Failure

    :group: utils

    """
    requests = import_module(
        "requests",
        pip_hint="pip install requests",
        conda_hint="conda install -c conda-forge requests",
    )

    # Resume download if file exists
    resume_header = {}
    mode = "wb"
    downloaded = 0
    name = Path(url).name
    if out_path.exists():
        downloaded = out_path.stat().st_size
        resume_header = {"Range": f"bytes={downloaded}-"}
        mode = "ab"
        msg = f"{name}: Resuming download from byte {downloaded}"
    else:
        msg = f"{name}: Starting download"

    try:
        with requests.get(url, stream=True, headers=resume_header, timeout=60) as r:
            if r.status_code == 416:
                return 0  # Already fully downloaded
            else:
                print(msg)
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                total += downloaded
                with open(out_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
    except Exception:
        return -1  # Indicate failure

    return 1  # Indicate success
