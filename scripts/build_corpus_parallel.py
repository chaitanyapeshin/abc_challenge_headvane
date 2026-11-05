from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--budget-gb", type=float, default=25.0)
    args = parser.parse_args()

    if not 1 <= args.workers <= 24:
        raise ValueError("Use between 1 and 24 workers.")

    reports = Path("reports/corpus")
    shards = Path("data/processed/corpus/users")
    logs = reports / "worker_logs"

    for directory in (reports, shards, logs):
        directory.mkdir(parents=True, exist_ok=True)

    # isolate each subprocess's raw files and download caches
    source = Path("scripts/build_corpus_managed.py").read_text()

    changes = {
        'RAW = Path("data/raw/corpus_transient")':
            'RAW = Path(os.environ["CORPUS_WORKER_RAW"])',
        '_download_root = Path("data/raw/corpus_transient").resolve()':
            '_download_root = Path(os.environ["CORPUS_WORKER_RAW"]).resolve()',
    }

    for old, new in changes.items():
        if source.count(old) != 1:
            raise RuntimeError(f"Expected exactly one match: {old}")
        source = source.replace(old, new, 1)

    worker_script = Path("scripts/corpus_worker.py")
    worker_script.write_text(source)

    pipeline_hash = hashlib.sha256(
        Path("scripts/preprocess_expanded_v2.py").read_bytes()
    ).hexdigest()

    # reuse the 500-user acquisition plan if it already exists
    plan_path = reports / f"plan_seed_{args.seed}.json"

    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        from huggingface_hub import HfApi

        api = HfApi()
        rng = random.Random(args.seed)

        print("Discovering candidate archives...", flush=True)
        directories = [
            entry.path
            for entry in api.list_repo_tree(
                repo_id="cschell/boxrr-23",
                repo_type="dataset",
                path_in_repo="users",
                recursive=False,
            )
            if entry.path.rsplit("/", 1)[-1].isdigit()
        ]
        rng.shuffle(directories)
        directories = directories[:20]

        pilot_users = {
            path.stem for path in Path("data/raw/pilot").rglob("*.tar")
        }

        queues = []
        for directory in directories:
            print(f"Listing {directory}...", flush=True)
            candidates = []

            for entry in api.list_repo_tree(
                repo_id="cschell/boxrr-23",
                repo_type="dataset",
                path_in_repo=directory,
                recursive=False,
            ):
                size = getattr(entry, "size", None)
                user = Path(entry.path).stem
                done = (
                    (reports / f"{user}.json").exists()
                    and (shards / f"{user}.h5").exists()
                )

                if (
                    entry.path.endswith(".tar")
                    and size is not None
                    and 0 < size <= 300_000_000
                    and user not in pilot_users
                    and not done
                ):
                    candidates.append({
                        "path": entry.path,
                        "size": int(size),
                    })

            rng.shuffle(candidates)
            queues.append(candidates)

        selected = []
        total = 0

        while len(selected) < args.users and any(queues):
            for queue in queues:
                if len(selected) >= args.users:
                    break
                while queue:
                    item = queue.pop()
                    if total + item["size"] <= args.budget_gb * 1_000_000_000:
                        selected.append(item)
                        total += item["size"]
                        break

        plan = {
            "settings": {
                "users": args.users,
                "chunks": 20,
                "budget_gb": args.budget_gb,
                "max_user_mb": 300.0,
                "seed": args.seed,
            },
            "archives": selected,
            "selected_bytes": total,
            "chunks": directories,
        }

        temporary = plan_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(plan, indent=2))
        os.replace(temporary, plan_path)

    # remove completed users from this run's work queue
    pending = []
    seen = set()

    for candidate in plan["archives"]:
        user = Path(candidate["path"]).stem
        if user in seen:
            continue
        seen.add(user)

        receipt = reports / f"{user}.json"
        output = shards / f"{user}.h5"

        if receipt.exists() and output.exists():
            record = json.loads(receipt.read_text())
            if record["pipeline_sha256"] != pipeline_hash:
                raise RuntimeError(f"Pipeline mismatch for {user}")
            continue

        pending.append(candidate)

    print(
        f"\nWorkers: {args.workers}; pending users: {len(pending)}",
        flush=True,
    )

    def run_user(candidate):
        user = Path(candidate["path"]).stem

        # stable per-user seed identifies its single-user acquisition plan
        worker_seed = int(hashlib.sha256(
            candidate["path"].encode()
        ).hexdigest()[:16], 16) + 1_000_000

        settings = {
            "users": 1,
            "chunks": 1,
            "budget_gb": 1.0,
            "max_user_mb": 300.0,
            "seed": worker_seed,
        }

        single_plan = {
            "settings": settings,
            "archives": [candidate],
            "selected_bytes": candidate["size"],
            "chunks": [str(Path(candidate["path"]).parent)],
        }

        single_path = reports / f"plan_seed_{worker_seed}.json"
        single_path.write_text(json.dumps(single_plan, indent=2))

        environment = os.environ.copy()
        environment["CORPUS_WORKER_RAW"] = str(
            Path("data/raw/corpus_parallel_tmp", user).resolve()
        )

        # prevent eight processes from each launching many numeric threads
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = "1"

        log = logs / f"{user}.log"

        with log.open("w") as handle:
            result = subprocess.run(
                [
                    sys.executable, str(worker_script),
                    "--users", "1",
                    "--chunks", "1",
                    "--budget-gb", "1",
                    "--max-user-mb", "300",
                    "--seed", str(worker_seed),
                ],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )

        receipt = reports / f"{user}.json"
        if result.returncode != 0 or not receipt.exists():
            raise RuntimeError(f"Worker failed or stopped for {user}; see {log}")

        return json.loads(receipt.read_text())["summary"]

    started = time.monotonic()
    completed = 0
    accepted = 0
    samples = 0
    failed = False
    cursor = 0
    active = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while cursor < len(pending) or active:
            while (
                not failed
                and cursor < len(pending)
                and len(active) < args.workers
            ):
                # reserve space for concurrent downloads and output writes
                reserve = 20_000_000_000 + args.workers * 600_000_000
                processed = sum(
                    path.stat().st_size for path in shards.glob("*.h5")
                )

                if (
                    shutil.disk_usage(shards).free < reserve
                    or processed >= 10_000_000_000
                ):
                    print(
                        "Storage limit reached; finishing active workers.",
                        flush=True,
                    )
                    failed = True
                    break

                candidate = pending[cursor]
                cursor += 1
                future = executor.submit(run_user, candidate)
                active[future] = candidate["path"]

            if not active:
                break

            finished, _ = wait(active, return_when=FIRST_COMPLETED)

            for future in finished:
                archive = active.pop(future)
                try:
                    summary = future.result()
                except Exception as exc:
                    print(f"\nERROR: {exc}", flush=True)
                    failed = True
                    continue

                completed += 1
                accepted += bool(summary["accepted_user_ids"])
                samples += summary["samples"]
                hours = max((time.monotonic() - started) / 3600, 1e-9)

                print(
                    f"[{completed}/{len(pending)}] {Path(archive).stem[:8]} "
                    f"| accepted users: {accepted} "
                    f"| motion: {samples / 360000:.2f} h "
                    f"| throughput: {completed / hours:.1f} users/h",
                    flush=True,
                )

            if failed and not active:
                break

    print("\nRUN SUMMARY")
    print(f"Newly completed users: {completed}")
    print(f"Newly completed users with accepted data: {accepted}")
    print(f"New samples: {samples:,}")
    print(f"Elapsed minutes: {(time.monotonic() - started) / 60:.1f}")
    print(f"Worker logs: {logs}")
    print("Rerun the same command to resume remaining work.")


if __name__ == "__main__":
    main()
