import os
import shutil
from pathlib import Path
import numpy as np

_download_root = Path("data/raw/corpus_transient").resolve()
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_CACHE"] = str(_download_root / "hub_cache")
os.environ["HF_XET_CACHE"] = str(_download_root / "xet_cache")

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import tarfile

import h5py
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from xror import XROR

from preprocess_expanded_v2 import process


REPO = "cschell/boxrr-23"
RAW = Path("data/raw/corpus_transient")
SHARDS = Path("data/processed/corpus/users")
REPORTS = Path("reports/corpus")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--budget-gb", type=float, default=3.0)
    parser.add_argument("--max-user-mb", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    if min(args.users, args.chunks) < 1:
        raise ValueError("User and chunk counts must be positive.")
    if min(args.budget_gb, args.max_user_mb) <= 0:
        raise ValueError("Download limits must be positive.")

    for directory in (RAW, SHARDS, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)

    pipeline_hash = hashlib.sha256(
        Path("scripts/preprocess_expanded_v2.py").read_bytes()
    ).hexdigest()

    with h5py.File("data/processed/expanded_v2.h5", "r") as pilot:
        config = json.loads(pilot.attrs["config"])

    # save a fixed acquisition plan so interrupted runs select the same users
    plan_path = REPORTS / f"plan_seed_{args.seed}.json"
    settings = vars(args)

    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["settings"] != settings:
            raise RuntimeError(
                "This seed already has a plan with different settings. "
                "Use a new seed for a new batch."
            )
    else:
        api = HfApi()
        rng = random.Random(args.seed)

        print("Discovering dataset chunks...", flush=True)
        directories = [
            entry.path
            for entry in api.list_repo_tree(
                repo_id=REPO,
                repo_type="dataset",
                path_in_repo="users",
                recursive=False,
            )
            if entry.path.rsplit("/", 1)[-1].isdigit()
        ]

        rng.shuffle(directories)
        directories = directories[:args.chunks]

        if not directories:
            raise RuntimeError("No dataset chunk directories found.")

        # exclude archives already acquired for the pilot
        existing_pilot = {
            path.stem
            for path in Path("data/raw/pilot").rglob("*.tar")
        }

        queues = []

        for directory in directories:
            print(f"Listing {directory}...", flush=True)
            candidates = []

            for entry in api.list_repo_tree(
                repo_id=REPO,
                repo_type="dataset",
                path_in_repo=directory,
                recursive=False,
            ):
                size = getattr(entry, "size", None)
                if (
                    entry.path.endswith(".tar")
                    and Path(entry.path).stem not in existing_pilot
                    and size is not None
                    and 0 < size <= args.max_user_mb * 1_000_000
                ):
                    candidates.append({
                        "path": entry.path,
                        "size": int(size),
                    })

            rng.shuffle(candidates)
            queues.append(candidates)

        # round-robin selection spreads the batch across sampled chunks
        selected = []
        total_bytes = 0
        budget = args.budget_gb * 1_000_000_000

        while len(selected) < args.users and any(queues):
            for queue in queues:
                if len(selected) >= args.users:
                    break

                while queue:
                    candidate = queue.pop()
                    if total_bytes + candidate["size"] <= budget:
                        selected.append(candidate)
                        total_bytes += candidate["size"]
                        break

        if not selected:
            raise RuntimeError("No archives fit the acquisition limits.")

        plan = {
            "settings": settings,
            "chunks": directories,
            "archives": selected,
            "selected_bytes": total_bytes,
        }
        atomic_json(plan_path, plan)

    print(
        f"\nBatch: {len(plan['archives'])} users, "
        f"{plan['selected_bytes'] / 1e9:.2f} GB of archives",
        flush=True,
    )

    summaries = []

    for candidate in plan["archives"]:
        archive_user = Path(candidate["path"]).stem
        output = SHARDS / f"{archive_user}.h5"
        receipt_path = REPORTS / f"{archive_user}.json"

        if receipt_path.exists() and output.exists():
            receipt = json.loads(receipt_path.read_text())

            if receipt["pipeline_sha256"] != pipeline_hash:
                raise RuntimeError(
                    f"Pipeline changed since processing {archive_user}. "
                    "Existing output was preserved."
                )

            with h5py.File(output, "r") as h5:
                if not bool(h5.attrs.get("complete", False)):
                    raise RuntimeError(f"Incomplete shard: {output}")

            print(f"Already processed: {archive_user}", flush=True)
            summaries.append(receipt["summary"])
            continue

        # check capacity before downloading the next user
        processed_bytes = sum(
            path.stat().st_size for path in SHARDS.glob("*.h5")
        )
        free_bytes = shutil.disk_usage(SHARDS).free

        if processed_bytes >= 10_000_000_000:
            print("\nStopping: processed corpus reached 10 GB.")
            break

        required_free = 20_000_000_000 + 2 * candidate["size"]
        if free_bytes < required_free:
            print("\nStopping: preserving at least 20 GB free space.")
            break

        # this dedicated directory contains only disposable download data
        if RAW.exists():
            shutil.rmtree(RAW)
        RAW.mkdir(parents=True, exist_ok=True)

        print(f"\nDownloading/reading {archive_user}...", flush=True)
        archive_path = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=candidate["path"],
            local_dir=str(RAW),
        )

        temporary = output.with_suffix(".h5.tmp")
        rows = []
        saved_segments = 0
        saved_samples = 0
        accepted_users = set()

        with h5py.File(temporary, "w") as h5:
            h5.attrs["complete"] = False
            h5.attrs["sample_rate_hz"] = 100
            h5.attrs["quaternion_order"] = "w,x,y,z"
            h5.attrs["config"] = json.dumps(config)
            h5.attrs["pipeline_sha256"] = pipeline_hash
            h5.attrs["source_archive"] = candidate["path"]
            root = h5.create_group("segments")

            with tarfile.open(archive_path, "r:*") as archive:
                members = [
                    member for member in archive.getmembers()
                    if member.isfile()
                    and member.name.lower().endswith(".xror")
                ]

                for member in tqdm(members, desc=archive_user[:8]):
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
                        row.update(
                            user_id=user_id,
                            session_id=session_id,
                        )

                        devices = data["info"]["hardware"]["devices"]
                        has_head = any(
                            str(device.get("type", "")).upper() == "HMD"
                            or str(device.get("joint", "")).upper() == "HEAD"
                            for device in devices
                        )

                        if not has_head:
                            row["status"] = "rejected_no_head_device"
                            rows.append(row)
                            continue

                        segments = process(data, row)

                    except Exception as exc:
                        row["status"] = "error"
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        rows.append(row)
                        continue

                    # storage failures stop the run rather than hiding data loss
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

                        saved_segments += 1
                        saved_samples += len(timestamps)
                        accepted_users.add(user_id)

                    rows.append(row)

            h5.attrs["complete"] = True

        # verify the written shard before publishing or deleting raw data
        with h5py.File(temporary, "r") as verified:
            assert bool(verified.attrs["complete"])
            verified_samples = [0]
            verified_segments = [0]

            def check_segment(name, obj):
                if not isinstance(obj, h5py.Group):
                    return
                if "quaternions" not in obj:
                    return

                t = obj["timestamps"][:]
                q = obj["quaternions"][:].astype(np.float64)

                assert q.shape == (len(t), 4), name
                assert len(t) >= 1000, name
                assert np.isfinite(q).all() and np.isfinite(t).all(), name
                assert np.allclose(
                    np.diff(t), 0.01, atol=1e-9, rtol=0
                ), name
                assert np.allclose(
                    np.linalg.norm(q, axis=1), 1, atol=1e-6
                ), name
                assert np.all(
                    np.sum(q[:-1] * q[1:], axis=1) >= -1e-7
                ), name

                verified_samples[0] += len(t)
                verified_segments[0] += 1

            verified["segments"].visititems(check_segment)
            assert verified_samples[0] == saved_samples
            assert verified_segments[0] == saved_segments

        os.replace(temporary, output)

        summary = {
            "archive_user": archive_user,
            "accepted_user_ids": sorted(accepted_users),
            "sessions": len(rows),
            "accepted_sessions": sum(
                row.get("status") == "accepted" for row in rows
            ),
            "segments": saved_segments,
            "samples": saved_samples,
            "outcomes": dict(Counter(row["status"] for row in rows)),
            "hdf5": output.as_posix(),
        }

        receipt = {
            "pipeline_sha256": pipeline_hash,
            "source_archive": candidate["path"],
            "summary": summary,
            "sessions": rows,
        }
        atomic_json(receipt_path, receipt)
        summaries.append(summary)

        # raw archive and its local download caches are now disposable
        shutil.rmtree(RAW)
        print("Verified output; deleted temporary raw download.", flush=True)

        print(
            f"Saved {saved_segments} segments, "
            f"{saved_samples:,} samples",
            flush=True,
        )

    accepted_ids = {
        user
        for summary in summaries
        for user in summary["accepted_user_ids"]
    }
    outcomes = Counter()
    for summary in summaries:
        outcomes.update(summary["outcomes"])

    print("\nBATCH SUMMARY")
    print(f"Processed archives: {len(summaries)}")
    print(f"Accepted users: {len(accepted_ids)}")
    print(f"Saved segments: {sum(s['segments'] for s in summaries)}")
    print(f"Saved samples: {sum(s['samples'] for s in summaries):,}")
    print("Session outcomes:")
    for status, count in outcomes.most_common():
        print(f"  {status}: {count}")

    print(f"\nAcquisition plan: {plan_path}")
    print(f"Per-user datasets: {SHARDS}")
    print(f"Per-user reports: {REPORTS}")


if __name__ == "__main__":
    main()
