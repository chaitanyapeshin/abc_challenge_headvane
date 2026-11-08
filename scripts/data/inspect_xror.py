from collections.abc import Mapping
from pathlib import Path
import tarfile

import numpy as np
from xror import XROR


def describe(value, path="recording", depth=0):
    # show structure and small previews without printing full recordings
    indent = "  " * depth

    if isinstance(value, np.ndarray):
        print(f"{indent}{path}: ndarray shape={value.shape}, dtype={value.dtype}")
        if value.ndim >= 1:
            print(f"{indent}  first rows: {value[:3]}")
        return

    if isinstance(value, (bytes, bytearray)):
        print(f"{indent}{path}: {type(value).__name__}, {len(value):,} bytes")
        return

    if isinstance(value, Mapping):
        print(f"{indent}{path}: dict keys={list(value.keys())}")
        if depth < 8:
            for key, item in value.items():
                describe(item, str(key), depth + 1)
        return

    if isinstance(value, (list, tuple)):
        print(f"{indent}{path}: {type(value).__name__}, length={len(value)}")
        if depth < 8:
            for index, item in enumerate(value[:5]):
                describe(item, f"[{index}]", depth + 1)
            if len(value) > 5:
                print(f"{indent}  ...")
        return

    if hasattr(value, "__dict__"):
        print(f"{indent}{path}: {type(value).__name__}")
        if depth < 8:
            describe(vars(value), "attributes", depth + 1)
        return

    preview = repr(value)
    print(f"{indent}{path}: {preview[:400]}")


archives = sorted(Path("data/raw/pilot").rglob("*.tar"))

if not archives:
    raise FileNotFoundError("No pilot .tar found in data/raw/pilot.")

archive_path = archives[0]
print(f"Archive: {archive_path}")

# read a recording directly from the archive without extracting files
with tarfile.open(archive_path, "r:*") as archive:
    member = next(
        (
            item for item in archive
            if item.isfile() and item.name.lower().endswith(".xror")
        ),
        None,
    )

    if member is None:
        raise RuntimeError("The archive contains no XROR recordings.")

    print(f"Recording: {member.name}")
    handle = archive.extractfile(member)

    if handle is None:
        raise RuntimeError("Could not read the selected recording.")

    with handle:
        recording = XROR.unpack(handle.read())

print("\nDecoded recording structure:\n")
describe(recording)
