import os
import sys

import numpy as np
import requests
from tqdm.auto import tqdm


def download_github_content(path: str, filename: str, chnksz: int = 1000):
    """
    Functions that downloads content from python_utils github
    repository.

    :param path: the file path
    :param filename: the filename
    :param chnksz: the chunk size
    """
    url = f"https://raw.githubusercontent.com/jpcano1/python_utils/main/{path}"

    try:
        r = requests.get(url, stream=True)
    except Exception as e:
        print(f"Error de conexión con el servidor: {e}")
        sys.exit()

    with open(filename, "wb") as f:
        try:
            total = int(np.ceil(int(r.headers.get("content-length")) / chnksz))
        except ArithmeticError:
            total = 0

        gen = r.iter_content(chunk_size=chnksz)

        for pkg in tqdm(gen, total=total, unit="KB"):
            f.write(pkg)

        f.close()
        r.close()
    return


def setup_general(dst: str = "utils") -> None:
    """
    Function that enables the general functions in google colab.
    """
    os.makedirs(dst, exist_ok=True)
    with open(f"{dst}/__init__.py", "wb") as f:
        f.close()

    download_github_content("utils/general.py", f"{dst}/general.py")
    print("General Functions Enabled Successfully")
