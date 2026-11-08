from pathlib import Path
import json
import tarfile
import warnings

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm
from xror import XROR


GAP_S = 0.100
MIN_SAMPLES = 1000
TARGET_HZ = 100
OUTPUT = Path("data/processed/pilot_clean.h5")


def extract_head(data):
    frames = np.asarray(data["frames"], dtype=np.float64)
    devices = data["info"]["hardware"]["devices"]
    offset = 1
    matches = []

    for device in devices:
        axes = list(device["axes"])

        if (
            str(device.get("type", "")).upper() == "HMD"
            or str(device.get("joint", "")).upper() == "HEAD"
        ):
            columns = {
                axis: offset + index
                for index, axis in enumerate(axes)
            }
            matches.append(columns)

        offset += len(axes)

    if len(matches) != 1:
        raise ValueError(f"Expected one head device; found {len(matches)}.")

    if frames.ndim != 2 or frames.shape[1] != offset:
        raise ValueError(f"Unexpected frame shape: {frames.shape}")

    columns = matches[0]

    # scipy uses vector-first quaternion order
    q = frames[:, [columns[a] for a in ("i", "j", "k", "1")]]
    return frames[:, 0], q


def align_signs(q):
    q = q.copy()

    if len(q) > 1:
        adjacent_dots = np.sum(q[:-1] * q[1:], axis=1)
        flips = np.where(adjacent_dots < 0, -1.0, 1.0)
        signs = np.concatenate(([1.0], np.cumprod(flips)))
        q *= signs[:, None]

    return q


def process(data, row):
    t, q = extract_head(data)
    row["raw_samples"] = len(t)

    # reject timestamp reversals rather than sorting motion into a new order
    finite_times = t[np.isfinite(t)]

    if np.any(np.diff(finite_times) < 0):
        row["status"] = "rejected_backward_timestamps"
        return []

    norms = np.linalg.norm(q, axis=1)
    valid = (
        np.isfinite(t)
        & np.isfinite(q).all(axis=1)
        & np.isfinite(norms)
        & (norms > 1e-8)
    )

    t = t[valid]
    q = q[valid]
    norms = norms[valid]

    if len(t) < 2:
        row["status"] = "rejected_too_few_valid_samples"
        return []

    q /= norms[:, None]
    session_start = float(t[0])

    # repeated timestamps are adjacent because reversals were rejected
    keep = np.concatenate((np.diff(t) != 0, [True]))
    row["duplicates_removed"] = int((~keep).sum())
    t, q = t[keep], q[keep]

    if len(t) < 2:
        row["status"] = "rejected_too_few_unique_samples"
        return []

    rate = float(1.0 / np.median(np.diff(t)))
    row["native_hz"] = rate

    if not 85 <= rate <= 110:
        row["status"] = "rejected_sampling_rate"
        return []

    # trim relative to the original session start
    trim_start = session_start + 5.0
    keep = t >= trim_start
    t, q = t[keep], q[keep]

    if len(t) < 20:
        row["status"] = "rejected_too_short_after_trim"
        return []

    q = align_signs(q)

    # flag abrupt raw orientation transitions before interpolation
    dots = np.abs(np.sum(q[:-1] * q[1:], axis=1))
    step_angles = np.degrees(2 * np.arccos(np.clip(dots, 0, 1)))
    step_speeds = step_angles / np.diff(t)
    jumps = (step_angles > 15.0) & (step_speeds > 1000.0)
    row["discontinuities"] = int(jumps.sum())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        euler = Rotation.from_quat(q).as_euler("xyz", degrees=True)

    sweeps = []
    velocities = []

    # use complete five-second windows anchored at the trim boundary
    window_count = int(np.floor((t[-1] - trim_start) / 5.0))

    for index in range(window_count):
        start = trim_start + 5.0 * index
        end = start + 5.0
        left = np.searchsorted(t, start, side="left")
        right = np.searchsorted(t, end, side="left")

        tw = t[left:right]
        angles = euler[left:right]

        if len(tw) < 10:
            continue

        # avoid scoring windows with long internal or boundary dropouts
        coverage = np.concatenate(([start], tw, [end]))

        if np.any(np.diff(coverage) > GAP_S + 1e-9):
            continue

        # exclude filter windows crossing a flagged transition
        if np.any(jumps[left:right - 1]):
            continue

        sweeps.append(
            np.percentile(angles, 90, axis=0)
            - np.percentile(angles, 10, axis=0)
        )

        diffs = np.abs(np.diff(angles, axis=0))
        wrapped = np.where(diffs > 180.0, 360.0 - diffs, diffs)
        velocities.append(wrapped.sum(axis=0) / 5.0)

    row["filter_windows"] = len(sweeps)

    if not sweeps:
        row["status"] = "rejected_no_filter_windows"
        return []

    sweep = float(np.max(np.median(sweeps, axis=0)))
    velocity = float(np.max(np.median(velocities, axis=0)))
    row["sweep_deg"] = sweep
    row["velocity_deg_s"] = velocity

    if not (sweep > 17.0 and velocity > 15.0):
        row["status"] = "rejected_motion"
        return []

    # interpolate only within segments separated by short enough intervals
    boundaries = np.flatnonzero((np.diff(t) > GAP_S) | jumps) + 1
    index_groups = np.split(np.arange(len(t)), boundaries)
    segments = []

    for indices in index_groups:
        if len(indices) < 2:
            continue

        ts = t[indices]
        qs = q[indices]

        count = int(np.floor((ts[-1] - ts[0]) * TARGET_HZ)) + 1

        if count < MIN_SAMPLES:
            continue

        grid = ts[0] + np.arange(count, dtype=np.float64) / TARGET_HZ
        grid = grid[grid <= ts[-1]]

        if len(grid) < MIN_SAMPLES:
            continue

        interpolated = Slerp(ts, Rotation.from_quat(qs))(grid).as_quat()
        interpolated = align_signs(interpolated)
        interpolated /= np.linalg.norm(interpolated, axis=1, keepdims=True)

        # convert scipy's x,y,z,w order into saved w,x,y,z order
        saved_q = interpolated[:, [3, 0, 1, 2]].astype(np.float32)

        if not np.isfinite(saved_q).all():
            raise ValueError("Nonfinite output quaternion.")

        if not np.allclose(np.linalg.norm(saved_q, axis=1), 1, atol=1e-6):
            raise ValueError("Output quaternion normalization failed.")

        if not np.allclose(np.diff(grid), 0.01, atol=1e-9, rtol=0):
            raise ValueError("Output timestamp spacing failed.")

        segments.append((grid, saved_q))

    row["segments_saved"] = len(segments)
    row["samples_saved"] = sum(len(grid) for grid, _ in segments)
    row["status"] = "accepted" if segments else "rejected_short_segments"
    return segments


def main():
    archives = sorted(Path("data/raw/pilot").rglob("*.tar"))

    if not archives:
        raise FileNotFoundError("No pilot archives found.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Rename it before rerunning."
        )

    rows = []

    with h5py.File(OUTPUT, "w") as h5:
        h5.attrs["sample_rate_hz"] = TARGET_HZ
        h5.attrs["quaternion_order"] = "w,x,y,z"
        h5.attrs["config"] = json.dumps({
            "jump_angle_deg": 15.0,
            "jump_speed_deg_s": 1000.0,
            "jump_policy": "split_before_interpolation_and_skip_filter_windows",
            "max_gap_s": GAP_S,
            "min_segment_samples": MIN_SAMPLES,
            "trim_s": 5,
            "filter_window_s": 5,
            "filter_window_policy": "complete_windows_without_long_gaps",
            "backward_timestamp_policy": "reject_session",
            "duplicate_timestamp_policy": "keep_last_valid_frame",
        })

        root = h5.create_group("segments")

        for archive_path in archives:
            with tarfile.open(archive_path, "r:*") as archive:
                members = [
                    m for m in archive.getmembers()
                    if m.isfile() and m.name.lower().endswith(".xror")
                ]

                for member in tqdm(members, desc="Preprocessing"):
                    row = {
                        "archive_member": member.name,
                        "segments_saved": 0,
                        "samples_saved": 0,
                    }

                    try:
                        handle = archive.extractfile(member)

                        if handle is None:
                            raise RuntimeError("Could not read recording.")

                        with handle:
                            data = XROR.unpack(handle.read()).data

                        user_id = str(data["info"]["user"]["id"])
                        session_id = str(data["$id"])
                        row.update(user_id=user_id, session_id=session_id)

                        segments = process(data, row)

                    except Exception as exc:
                        row["status"] = "error"
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        rows.append(row)
                        continue

                    # storage errors stop the run rather than being hidden
                    for index, (timestamps, quaternions) in enumerate(segments):
                        group = root.create_group(
                            f"{user_id}/{session_id}/{index:03d}"
                        )
                        group.attrs["user_id"] = user_id
                        group.attrs["session_id"] = session_id
                        group.attrs["source_member"] = member.name

                        group.create_dataset(
                            "timestamps",
                            data=timestamps,
                            compression="gzip",
                            compression_opts=1,
                        )
                        group.create_dataset(
                            "quaternions",
                            data=quaternions,
                            compression="gzip",
                            compression_opts=1,
                        )

                    rows.append(row)

        report = pd.DataFrame(rows)
        h5.attrs["complete"] = True

    report.to_csv("reports/pilot_clean_preprocessing.csv", index=False)

    print("\nSession outcomes:")
    print(report["status"].value_counts().to_string())
    print(f"\nSaved segments: {int(report['segments_saved'].sum())}")
    print(f"Saved samples: {int(report['samples_saved'].sum()):,}")
    print(f"Output size: {OUTPUT.stat().st_size / 1024**2:.2f} MiB")
    print(f"\nHDF5: {OUTPUT}")
    print("Report: reports/pilot_clean_preprocessing.csv")

    errors = report[report["status"] == "error"]

    if not errors.empty:
        print("\nErrors:")
        print(errors[["archive_member", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
