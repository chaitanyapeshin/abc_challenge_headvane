from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


report = pd.read_csv("reports/expanded_v2_preprocessing.csv")
accepted = report.loc[report["status"] == "accepted"]

print("Accepted sessions by user:")
print(accepted.groupby("user_id").size().to_string())
print(f"\nAccepted users: {accepted['user_id'].nunique()}")

rows = []
preview = None

with h5py.File("data/processed/expanded_v2.h5", "r") as h5:
    assert bool(h5.attrs.get("complete", False)), "Incomplete HDF5."
    assert h5.attrs["quaternion_order"] == "w,x,y,z"

    def check(name, obj):
        global preview

        if not isinstance(obj, h5py.Group) or "quaternions" not in obj:
            return

        q = obj["quaternions"][:]
        t = obj["timestamps"][:]

        assert q.ndim == 2 and q.shape[1] == 4, name
        assert len(t) == len(q) and len(t) >= 1000, name
        assert q.dtype == np.float32, name
        assert np.isfinite(q).all() and np.isfinite(t).all(), name
        assert np.allclose(np.diff(t), 0.01, atol=1e-9, rtol=0), name

        norms = np.linalg.norm(q.astype(np.float64), axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6), name

        q64 = q.astype(np.float64)
        dots = np.sum(q64[:-1] * q64[1:], axis=1)
        assert np.all(dots >= -1e-7), name

        # use quaternion differences for a wrap-free motion diagnostic
        unit_q = q64 / norms[:, None]
        relative_dots = np.abs(
            np.sum(unit_q[:-1] * unit_q[1:], axis=1)
        )
        speeds = (
            np.degrees(2 * np.arccos(np.clip(relative_dots, 0, 1)))
            / np.diff(t)
        )

        rows.append({
            "segment": name,
            "user_id": obj.attrs["user_id"],
            "session_id": obj.attrs["session_id"],
            "samples": len(q),
            "duration_s": float(t[-1] - t[0]),
            "windows_400_20": max(0, len(q) - 420 + 1),
            "max_norm_error": float(np.max(np.abs(norms - 1))),
            "speed_p99_deg_s": float(np.percentile(speeds, 99)),
            "speed_max_deg_s": float(speeds.max()),
        })

        if preview is None:
            preview = (name, t, q, speeds)

    h5["segments"].visititems(check)

verified = pd.DataFrame(rows)
assert not verified.empty, "No segments found."

assert int(verified["samples"].sum()) == int(
    accepted["samples_saved"].sum()
), "Sample totals disagree."

assert len(verified) == int(
    accepted["segments_saved"].sum()
), "Segment totals disagree."

Path("reports").mkdir(exist_ok=True)
verified.to_csv("reports/expanded_v2_verified_segments.csv", index=False)

print("\nVerified segments:")
print(verified.to_string(index=False))
print(f"\nTotal samples: {verified['samples'].sum():,}")
print(
    "Potential stride-1 windows:",
    f"{verified['windows_400_20'].sum():,}",
)
print("\nAll structural checks passed.")

name, t, q, speeds = preview
elapsed = t - t[0]
mask = elapsed <= 20
speed_mask = elapsed[1:] <= 20

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

for index, label in enumerate(["w", "x", "y", "z"]):
    axes[0].plot(elapsed[mask], q[mask, index], label=label)

axes[0].set_ylabel("Quaternion component")
axes[0].legend(ncol=4)
axes[0].set_title("First saved segment: first 20 seconds")

axes[1].plot(elapsed[1:][speed_mask], speeds[speed_mask])
axes[1].set_ylabel("Angular speed (deg/s)")
axes[1].set_xlabel("Elapsed time (s)")

fig.tight_layout()
fig.savefig("reports/expanded_v2_preview.png", dpi=150)
plt.close(fig)

print("Saved reports/expanded_v2_verified_segments.csv")
print("Saved reports/expanded_v2_preview.png")
