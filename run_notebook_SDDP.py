#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import time
from pathlib import Path

NOTEBOOKS = [
    "Wind_Ind_NC.ipynb",
    "Wind_Ind_OH.ipynb",
    "Wind_DD_NC.ipynb",
    "Wind_DD_OH.ipynb",
    "Wind_DRO_NC.ipynb",
    "Wind_DRO_OH.ipynb",
    "Wind_GMMDRO_NC.ipynb",
    "Wind_GMMDRO_OH.ipynb",
]

KERNEL_NAME = None           
TIMEOUT_SEC = 0
INPLACE = True
STOP_ON_ERROR = True

def run_nb(notebook: Path) -> int:
    args = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout={}".format(TIMEOUT_SEC if TIMEOUT_SEC > 0 else -1),
    ]
    if INPLACE:
        args.append("--inplace")
    else:
        args += ["--output", f"{notebook.stem}_executed.ipynb"]

    if KERNEL_NAME:
        args += ["--ExecutePreprocessor.kernel_name={}".format(KERNEL_NAME)]

    args.append(str(notebook))

    print(f"\n=== Running: {notebook.name} ===")
    t0 = time.time()
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0

    log_path = notebook.with_suffix(".runlog.txt")
    log_path.write_text(proc.stdout or "")

    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"--- {notebook.name}: {status} | {dt:.1f}s | log -> {log_path.name}")
    return proc.returncode

def main():
    root = Path.cwd()
    missing = [nb for nb in NOTEBOOKS if not (root / nb).exists()]
    if missing:
        print("No file:", ", ".join(missing))
        sys.exit(1)

    for nb in NOTEBOOKS:
        code = run_nb(Path(nb))
        if code != 0 and STOP_ON_ERROR:
            print("\n(STOP_ON_ERROR=True)")
            sys.exit(code)

    print("\n Finished")

if __name__ == "__main__":
    main()
