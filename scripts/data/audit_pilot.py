from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
from tqdm import tqdm
from xror import XROR


def audit_recording(data):
    devices = data["info"]["hardware"]["devices"]

    # locate headset columns using device metadata
    offset = 1
    head_columns = None
    head_axes = None

    for device in devices:
        axes = list(device["axes"])

        if (
            str(device.get("type", "")).upper() == "HMD"
            or str(device.get("joint", "")).upper() == "HEAD"
        ):
            if head_columns is not None:
                raise ValueError("Multiple head devices found.")

            head_axes = axes
            head_columns = {
                axis: offset + index
                for index, axis in enumerate(axes)
            }

        offset += len(axes)

    if head_columns is None:
        raise ValueError("No head device found.")

    frames = np.asarray(data["frames"], dtype=np.float64)

    if frames.ndim != 2 or frames.shape[1] != offset:
        raise ValueError(
            f"Unexpected frame shape {frames.shape}; expected {offset} columns."
        )

    # inspect native timestamps before removing duplicate samples
    timestamps = frames[:, 0]
    finite_time = np.isfinite(timestamps)
    times = timestamps[finite_time]
    intervals = np.diff(times)
    positive_intervals = intervals[intervals > 0]

    median_dt = (
        float(np.median(positive_intervals))
        if len(positive_intervals) else np.nan
    )
    native_hz = 1.0 / median_dt if median_dt > 0 else np.nan

    result = {
        "user_id": str(data["info"].get("user", {}).get("id", "")),
        "session_id": str(data.get("$id", "")),
        "frames": len(frames),
        "head_axes": ",".join(head_axes),
        "duration_s": float(times[-1] - times[0]) if len(times) else np.nan,
        "invalid_timestamps": int((~finite_time).sum()),
        "zero_intervals": int(np.sum(intervals == 0)),
        "backward_intervals": int(np.sum(intervals < 0)),
        "median_positive_dt_s": median_dt,
        "native_hz_estimate": native_hz,
        "rate_in_band": bool(85 <= native_hz <= 110),
        "gaps_over_100ms": int(np.sum(intervals > 0.1)),
        "max_positive_gap_s": (
            float(positive_intervals.max())
            if len(positive_intervals) else np.nan
        ),
    }

    # check quaternion quality without assuming an undocumented column order
    if all(axis in head_columns for axis in ("i", "j", "k", "1")):
        quaternion_columns = [
            head_columns[axis] for axis in ("1", "i", "j", "k")
        ]
        quaternions = frames[:, quaternion_columns]
        finite_q = np.isfinite(quaternions).all(axis=1)
        norms = np.linalg.norm(quaternions[finite_q], axis=1)

        result.update({
            "quaternion_columns_wijk": str(quaternion_columns),
            "invalid_quaternions": int((~finite_q).sum()),
            "near_zero_quaternions": int(np.sum(norms < 1e-8)),
            "median_quaternion_norm": (
                float(np.median(norms)) if len(norms) else np.nan
            ),
        })
    else:
        result["quaternion_columns_wijk"] = "Unrecognized axis labels"

    return result


def main():
    archives = sorted(Path("data/raw/pilot").rglob("*.tar"))

    if not archives:
        raise FileNotFoundError("No pilot archives found.")

    rows = []
    errors = []

    for archive_path in archives:
        with tarfile.open(archive_path, "r:*") as archive:
            members = [
                member for member in archive.getmembers()
                if member.isfile() and member.name.lower().endswith(".xror")
            ]

            for member in tqdm(members, desc=archive_path.stem[:8]):
                try:
                    handle = archive.extractfile(member)

                    if handle is None:
                        raise RuntimeError("Could not read archive member.")

                    with handle:
                        data = XROR.unpack(handle.read()).data

                    row = audit_recording(data)
                    row["archive_member"] = member.name
                    rows.append(row)

                except Exception as exc:
                    errors.append({
                        "archive_member": member.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    Path("reports").mkdir(exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv("reports/pilot_audit.csv", index=False)
    pd.DataFrame(
        errors, columns=["archive_member", "error"]
    ).to_csv("reports/pilot_audit_errors.csv", index=False)

    print(f"\nSuccessfully audited: {len(rows)}")
    print(f"Failed: {len(errors)}")

    if not report.empty:
        print("\nHead axis layouts:")
        print(report["head_axes"].value_counts().to_string())

        print("\nNative sampling rate estimates:")
        print(report["native_hz_estimate"].describe().to_string())

        print(
            "\nSessions with estimated rate within 85–110 Hz:",
            int(report["rate_in_band"].sum()),
        )

        print("\nFirst five sessions:")
        columns = [
            "session_id",
            "frames",
            "duration_s",
            "native_hz_estimate",
            "zero_intervals",
            "backward_intervals",
            "quaternion_columns_wijk",
            "median_quaternion_norm",
        ]
        print(report.reindex(columns=columns).head().to_string(index=False))

    if errors:
        print("\nFirst three errors:")
        for error in errors[:3]:
            print(error)

    print("\nSaved reports/pilot_audit.csv")
    print("Saved reports/pilot_audit_errors.csv")


if __name__ == "__main__":
    main()
