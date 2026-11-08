from pathlib import Path
import tarfile

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from xror import XROR

from preprocess_pilot import extract_head


SEGMENT = (
    "0438a986-434b-4f45-80a5-4bbb018d2cb8/"
    "68034c31-30cd-4b19-9da6-13a7f78d2b8c/000"
)


def motion_steps(t, q):
    q = q.astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    dots = np.abs(np.sum(q[:-1] * q[1:], axis=1))
    angles = np.degrees(2 * np.arccos(np.clip(dots, 0, 1)))
    dt = np.diff(t)

    speed = np.full(len(dt), np.nan)
    np.divide(angles, dt, out=speed, where=dt > 0)

    return angles, dt, speed


with h5py.File("data/processed/pilot.h5", "r") as h5:
    group = h5["segments"][SEGMENT]
    t = group["timestamps"][:]
    q = group["quaternions"][:]
    source_member = group.attrs["source_member"]

angles, dt, speed = motion_steps(t, q)
peak = int(np.nanargmax(speed))
peak_time = float((t[peak] + t[peak + 1]) / 2)

print(f"Segment: {SEGMENT}")
print(f"Peak interval: {t[peak]:.6f} to {t[peak + 1]:.6f} s")
print(f"Rotation during interval: {angles[peak]:.4f} degrees")
print(f"Interval duration: {dt[peak]:.6f} s")
print(f"Angular speed: {speed[peak]:.2f} degrees/s")

raw_t = None
raw_q = None

for archive_path in sorted(Path("data/raw/pilot").rglob("*.tar")):
    with tarfile.open(archive_path, "r:*") as archive:
        try:
            member = archive.getmember(source_member)
        except KeyError:
            continue

        handle = archive.extractfile(member)

        if handle is None:
            raise RuntimeError("Could not read original recording.")

        with handle:
            data = XROR.unpack(handle.read()).data

        raw_t, raw_q = extract_head(data)
        break

if raw_t is None:
    raise FileNotFoundError("Original recording was not found.")

norms = np.linalg.norm(raw_q, axis=1)
valid = (
    np.isfinite(raw_t)
    & np.isfinite(raw_q).all(axis=1)
    & np.isfinite(norms)
    & (norms > 1e-8)
)
raw_t, raw_q = raw_t[valid], raw_q[valid]

# match the preprocessing policy for duplicate timestamps
keep = np.concatenate((np.diff(raw_t) != 0, [True]))
raw_t, raw_q = raw_t[keep], raw_q[keep]

raw_angles, raw_dt, raw_speed = motion_steps(raw_t, raw_q)
raw_mid = (raw_t[:-1] + raw_t[1:]) / 2
saved_mid = (t[:-1] + t[1:]) / 2

near = np.abs(raw_mid - peak_time) <= 0.15

print("\nOriginal recording intervals near the peak:")
print("start_s       end_s         dt_s       angle_deg    speed_deg_s")
for index in np.flatnonzero(near):
    print(
        f"{raw_t[index]:12.6f} "
        f"{raw_t[index + 1]:12.6f} "
        f"{raw_dt[index]:10.6f} "
        f"{raw_angles[index]:12.4f} "
        f"{raw_speed[index]:12.2f}"
    )

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

for axis, half_width in zip(axes, [1.0, 0.15]):
    raw_mask = np.abs(raw_mid - peak_time) <= half_width
    saved_mask = np.abs(saved_mid - peak_time) <= half_width

    axis.plot(
        raw_mid[raw_mask] - peak_time,
        raw_speed[raw_mask],
        "o-",
        label="Original",
    )
    axis.plot(
        saved_mid[saved_mask] - peak_time,
        speed[saved_mask],
        ".-",
        label="Resampled 100 Hz",
    )
    axis.set_ylabel("Angular speed (deg/s)")
    axis.legend()
    axis.grid(alpha=0.3)

axes[0].set_title("Largest speed spike: original versus resampled")
axes[1].set_xlabel("Time relative to peak (s)")
fig.tight_layout()

output = Path("reports/pilot_spike_diagnostic.png")
fig.savefig(output, dpi=150)
plt.close(fig)

print(f"\nSaved {output}")
