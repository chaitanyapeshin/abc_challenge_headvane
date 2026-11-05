"""Full BOXRR acquisition; run from the abc-headpose-challenge repository root.

Reuses build_corpus_managed.py's storage/verification code and unchanged
preprocess_expanded_v2.py. No user, chunk, archive-size or transfer-budget cap.
"""
from pathlib import Path, PurePosixPath
from collections import Counter
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

REPO = "cschell/boxrr-23"
REPORTS = Path("reports/corpus")
STATE = REPORTS / "full_dataset"
SHARDS = Path("data/processed/corpus/users")
RAW = Path("data/raw/corpus_full_tmp")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Managed script differs from expected source: {old[:120]}")
    return source.replace(old, new, 1)


def make_worker(source):
    # reuse the existing worker instead of duplicating the cleaning algorithm
    source = replace_once(source,
        '_download_root = Path("data/raw/corpus_transient").resolve()',
        '_download_root = Path(os.environ["CORPUS_WORKER_RAW"]).resolve()')
    source = replace_once(source,
        'RAW = Path("data/raw/corpus_transient")',
        'RAW = Path(os.environ["CORPUS_WORKER_RAW"])')
    start = source.index('    # save a fixed acquisition plan')
    end = source.index('    print(\n        f"\\nBatch:', start)
    source = source[:start] + '''    plan_path = Path(os.environ["CORPUS_SINGLE_PLAN"])
    plan = json.loads(plan_path.read_text())

''' + source[end:]
    source = replace_once(source,
        '        if processed_bytes >= 10_000_000_000:',
        '        if False:  # parent enforces the configured corpus limit')
    source = replace_once(source,
        '            local_dir=str(RAW),',
        '            local_dir=str(RAW),\n            revision=os.environ["CORPUS_REVISION"],')
    # old receipts with recording errors must be retried, not skipped
    source = replace_once(source,
        '        if receipt_path.exists() and output.exists():',
        '''        if (receipt_path.exists() and output.exists()
                and not json.loads(receipt_path.read_text())["summary"]
                    .get("outcomes", {}).get("error", 0)):''')
    # release the decoded recording before decoding the next one
    source = replace_once(source, '                    rows.append(row)\n\n            h5.attrs',
        '                    rows.append(row)\n                    del data, segments\n\n            h5.attrs')
    compile(source, "full_worker.py", "exec")
    return source


def discover(api):
    plan_path = STATE / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if not plan.get("discovery_complete"):
            raise RuntimeError("Plan was not finalized.")
        return plan
    # pin both metadata and downloads to one dataset snapshot
    revision_path = STATE / "revision.json"
    if revision_path.exists():
        revision = json.loads(revision_path.read_text())["revision"]
    else:
        revision = api.repo_info(repo_id=REPO, repo_type="dataset").sha
        atomic_json(revision_path, {"revision": revision})
    chunks = STATE / "discovery"
    chunks.mkdir(exist_ok=True)
    entries = []
    # recursive discovery includes every directory under users, without numeric assumptions
    # completed top-level directory listings are checkpointed independently
    roots = list(api.list_repo_tree(repo_id=REPO, repo_type="dataset",
        revision=revision, path_in_repo="users", recursive=False))
    for root in roots:
        if getattr(root, "size", None) is not None:
            if root.path.endswith(".tar"):
                entries.append({"path": root.path, "size": int(root.size)})
            continue
        cache = chunks / (hashlib.sha256(root.path.encode()).hexdigest() + ".json")
        if cache.exists():
            items = json.loads(cache.read_text())
        else:
            print(f"Discovering {root.path}...", flush=True)
            items = []
            for entry in api.list_repo_tree(repo_id=REPO, repo_type="dataset",
                    revision=revision, path_in_repo=root.path, recursive=True):
                if entry.path.endswith(".tar"):
                    size = getattr(entry, "size", None)
                    if size is None or size <= 0:
                        raise RuntimeError(f"Missing/invalid size: {entry.path}")
                    items.append({"path": entry.path, "size": int(size)})
            atomic_json(cache, items)
        entries.extend(items)
    seen = set()
    for item in entries:
        p = PurePosixPath(item["path"])
        if p.is_absolute() or ".." in p.parts or p.parts[0] != "users":
            raise RuntimeError(f"Unexpected archive path: {p}")
        if p.stem in seen:
            raise RuntimeError(f"Duplicate archive user ID: {p.stem}")
        seen.add(p.stem)
    if not entries:
        raise RuntimeError("No user archives discovered.")
    plan = {"repo": REPO, "revision": revision, "discovery_complete": True,
            "archives": sorted(entries, key=lambda x: x["path"]),
            "archive_bytes": sum(x["size"] for x in entries)}
    atomic_json(plan_path, plan)
    return plan


def inspect_existing(candidate, pipeline_hash, h5py):
    user = Path(candidate["path"]).stem
    receipt_path = REPORTS / f"{user}.json"
    output = SHARDS / f"{user}.h5"
    if not receipt_path.exists() or not output.exists():
        return False
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("pipeline_sha256") != pipeline_hash:
        raise RuntimeError(f"Pipeline mismatch for {user}; preserved existing files.")
    if receipt.get("source_archive") != candidate["path"]:
        raise RuntimeError(f"Archive mismatch for {user}.")
    if receipt["summary"].get("outcomes", {}).get("error", 0):
        return False
    with h5py.File(output, "r") as h5:
        if not bool(h5.attrs.get("complete", False)):
            raise RuntimeError(f"Incomplete existing shard: {output}")
        if h5.attrs.get("pipeline_sha256") != pipeline_hash:
            raise RuntimeError(f"Shard pipeline mismatch: {output}")
        if h5.attrs.get("source_archive") != candidate["path"]:
            raise RuntimeError(f"Shard source mismatch: {output}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-free-gb", type=float, default=20)
    parser.add_argument("--max-processed-gb", type=float, default=0,
        help="0 means no corpus-size cap; free-space checks remain enabled")
    parser.add_argument("--disk-factor", type=float, default=3,
        help="reserve this multiple of each active archive's compressed size")
    parser.add_argument("--worker-headroom-gb", type=float, default=1,
        help="extra output/disk headroom per active worker (not a RAM limit)")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 36 or args.min_free_gb < 20:
        parser.error("Use 1–36 workers and at least 20 GB free-space reserve.")
    if args.max_processed_gb < 0 or args.disk_factor < 2 or args.worker_headroom_gb < 0:
        parser.error("Invalid storage settings.")
    if sys.flags.optimize:
        parser.error("Do not use python -O: worker verification uses assertions.")
    for directory in (STATE, SHARDS, RAW):
        directory.mkdir(parents=True, exist_ok=True)
    # macOS/Linux advisory lock prevents two full runners from sharing outputs
    lock = (STATE / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another full-dataset runner is active.")
    import h5py
    from huggingface_hub import HfApi
    pipeline_hash = fingerprint(Path("scripts/preprocess_expanded_v2.py"))
    managed = Path("scripts/build_corpus_managed.py")
    source = make_worker(managed.read_text())
    identity = {"pipeline_sha256": pipeline_hash, "managed_sha256": fingerprint(managed)}
    identity_path = STATE / "pipeline.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("Preprocessing or managed script changed since this full run began.")
    atomic_json(identity_path, identity)
    worker = STATE / "full_worker.py"
    worker.write_text(source)
    plan = discover(HfApi())
    print(f"Snapshot: {plan['revision']}\nDiscovered archives: {len(plan['archives']):,}"
          f"\nTotal source archives: {plan['archive_bytes']/1e9:,.2f} GB", flush=True)
    pending = []
    completed = {}
    for i, candidate in enumerate(plan["archives"]):
        user = Path(candidate["path"]).stem
        if inspect_existing(candidate, pipeline_hash, h5py):
            completed[user] = candidate["path"]
        else:
            pending.append(candidate)
        if (i + 1) % 5000 == 0:
            print(f"Checked existing outputs: {i+1:,}", flush=True)
    print(f"Already completed: {len(completed):,}; pending: {len(pending):,}", flush=True)
    # old receipt-based completion is reused; the pilot is deliberately reprocessed
    # into per-user shards so this output tree is self-contained
    initial = len(completed)
    attempted = {}
    failures = {}
    active = {}
    logs = STATE / "logs"
    jobs = STATE / "jobs"
    logs.mkdir(exist_ok=True)
    jobs.mkdir(exist_ok=True)
    cursor = 0
    processed_bytes = sum(p.stat().st_size for p in SHARDS.glob("*.h5"))
    started = time.monotonic()
    stopping = args.audit_only
    reason = "audit only" if stopping else "queue exhausted"

    def save_audit():
        atomic_json(STATE / "audit.json", {
            "revision": plan["revision"], "pipeline_sha256": pipeline_hash,
            "discovered": len(plan["archives"]), "completed": len(completed),
            "remaining": len(plan["archives"]) - len(completed),
            "completed_before_run": initial, "newly_completed": len(completed)-initial,
            "failed_this_run": failures, "attempted_this_run": attempted,
            "stop_reason": reason,
            "elapsed_minutes": (time.monotonic()-started)/60,
            "note": "Pending paths = plan archives minus completed_archive_users.",
            "completed_archive_users": sorted(completed),
        })

    def launch(candidate):
        user = Path(candidate["path"]).stem
        job = jobs / f"{user}.json"
        atomic_json(job, {"archives": [candidate], "selected_bytes": candidate["size"]})
        environment = os.environ.copy()
        environment["CORPUS_WORKER_RAW"] = str((RAW / user).resolve())
        environment["CORPUS_SINGLE_PLAN"] = str(job.resolve())
        environment["CORPUS_REVISION"] = plan["revision"]
        environment["PYTHONPATH"] = str(Path("scripts").resolve()) + os.pathsep + environment.get("PYTHONPATH", "")
        environment.pop("PYTHONOPTIMIZE", None)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        log_path = logs / f"{user}.log"
        previous_size = (SHARDS / f"{user}.h5").stat().st_size if (SHARDS / f"{user}.h5").exists() else 0
        handle = log_path.open("w")
        try:
            # separate process group lets Ctrl+C drain rather than kill workers mid-write
            process = subprocess.Popen([sys.executable, "-u", str(worker.resolve())],
                env=environment, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True)
        finally:
            handle.close()
        active[user] = (process, candidate, previous_size)
        attempted[user] = candidate["path"]

    save_audit()
    last_heartbeat = time.monotonic()
    last_audit = time.monotonic()
    while active or (cursor < len(pending) and not stopping):
        try:
            while not stopping and cursor < len(pending) and len(active) < args.workers:
                candidate = pending[cursor]
                # reserve conservatively for all active jobs, even if partly downloaded
                reserved = sum(args.disk_factor*c["size"] + args.worker_headroom_gb*1e9
                               for _, c, _ in active.values())
                required = args.min_free_gb*1e9 + reserved + args.disk_factor*candidate["size"] + args.worker_headroom_gb*1e9
                limit = args.max_processed_gb*1e9
                if limit and processed_bytes >= limit:
                    stopping, reason = True, "processed corpus limit"
                    break
                if shutil.disk_usage(SHARDS).free < required:
                    if active:
                        break
                    stopping, reason = True, f"insufficient disk space for {candidate['path']}"
                    print(reason, flush=True)
                    break
                launch(candidate)
                cursor += 1
            finished = [u for u, (p, _, _) in active.items() if p.poll() is not None]
            for user in finished:
                process, candidate, previous_size = active.pop(user)
                try:
                    if process.returncode != 0:
                        raise RuntimeError(f"Worker exit {process.returncode}")
                    if not inspect_existing(candidate, pipeline_hash, h5py):
                        raise RuntimeError("Missing output or recording-level errors in receipt")
                    completed[user] = candidate["path"]
                except Exception as exc:
                    failures[user] = {"path": candidate["path"], "error": str(exc),
                                      "log": str(logs / f"{user}.log")}
                    print(f"FAILED {user}: {exc}; continuing", flush=True)
                finally:
                    output = SHARDS / f"{user}.h5"
                    processed_bytes += (output.stat().st_size if output.exists() else 0) - previous_size
                    # remove only this runner's disposable per-user cache, also on failure
                    disposable = RAW / user
                    if disposable.exists():
                        shutil.rmtree(disposable)
                elapsed = max((time.monotonic()-started)/3600, 1e-9)
                print(f"Completed {len(completed):,}/{len(plan['archives']):,} | "
                      f"new {len(completed)-initial:,} | failures {len(failures):,} | "
                      f"{len(attempted)/elapsed:,.0f} attempts/h", flush=True)
            if finished and time.monotonic() - last_audit >= 30:
                save_audit()
                last_audit = time.monotonic()
            if time.monotonic()-last_heartbeat >= 30:
                print(f"Active workers: {len(active)}; pending unscheduled: {len(pending)-cursor:,}; "
                      f"free disk: {shutil.disk_usage(SHARDS).free/1e9:.1f} GB", flush=True)
                last_heartbeat = time.monotonic()
            if active:
                time.sleep(0.5)
        except KeyboardInterrupt:
            stopping, reason = True, "interrupted; active workers drained"
            print("Stopping new jobs; waiting for active workers to finish. Keep this terminal open.", flush=True)
    save_audit()
    print(f"\nFULL DATASET SUMMARY\nDiscovered: {len(plan['archives']):,}"
          f"\nCompleted: {len(completed):,}\nRemaining: {len(plan['archives'])-len(completed):,}"
          f"\nFailed this run: {len(failures):,}\nStop reason: {reason}"
          f"\nAudit: {STATE / 'audit.json'}", flush=True)
    if failures:
        print("Rerun the same command to retry failed and pending archives.", flush=True)
    lock.close()
    return 0 if len(completed) == len(plan["archives"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
