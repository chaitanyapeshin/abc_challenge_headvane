from pathlib import Path
import json
import tarfile

import numpy as np
import pandas as pd
from xror import XROR


report = pd.read_csv("reports/expanded_clean_preprocessing.csv")

# inspect one unrecognized-device recording per user
errors = report.loc[report["status"] == "error"].copy()
errors["archive_user"] = errors["archive_member"].str.split("/").str[0]
examples = errors.groupby("archive_user", sort=True).head(1)

device_targets = set(examples["archive_member"])
backward_targets = set(
    report.loc[
        report["status"] == "rejected_backward_timestamps",
        "archive_member",
    ]
)

device_rows = []
backward_rows = []

for archive_path in sorted(Path("data/raw/pilot").rglob("*.tar")):
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            inspect_devices = member.name in device_targets
            inspect_backwards = member.name in backward_targets

            if not member.isfile() or not (
                inspect_devices or inspect_backwards
            ):
                continue

            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"Cannot read {member.name}")

            with handle:
                data = XROR.unpack(handle.read()).data

            if inspect_devices:
                devices = data.get("info", {}).get(
                    "hardware", {}
                ).get("devices", [])

                device_rows.append({
                    "archive_member": member.name,
                    "devices": devices,
                })

            if inspect_backwards:
                frames = np.asarray(data["frames"], dtype=np.float64)
                t = frames[:, 0]
                t = t[np.isfinite(t)]
                dt = np.diff(t)
                indices = np.flatnonzero(dt < 0)

                backward_rows.append({
                    "archive_member": member.name,
                    "backward_steps": len(indices),
                    "largest_reversal_s": float((-dt[indices]).max()),
                    "first_reversal_start_s": float(t[indices[0]]),
                    "first_reversal_end_s": float(t[indices[0] + 1]),
                    "first_reversal_frame": int(indices[0]),
                })

print("DEVICE METADATA FOR UNRECOGNIZED RECORDINGS")
print(json.dumps(device_rows, indent=2, default=str))

backwards = pd.DataFrame(backward_rows)

print("\nBACKWARD TIMESTAMP SUMMARY")
if not backwards.empty:
    print(
        backwards[
            ["backward_steps", "largest_reversal_s"]
        ].describe().to_string()
    )

    print("\nTen largest reversals:")
    print(
        backwards.sort_values(
            "largest_reversal_s", ascending=False
        ).head(10).to_string(index=False)
    )

Path("reports/rejected_device_examples.json").write_text(
    json.dumps(device_rows, indent=2, default=str)
)
backwards.to_csv("reports/backward_timestamp_details.csv", index=False)

print("\nSaved diagnostic reports.")
