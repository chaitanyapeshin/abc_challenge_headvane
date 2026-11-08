from pathlib import Path
import random

from huggingface_hub import HfApi, hf_hub_download


REPO = "cschell/boxrr-23"
DESTINATION = Path("data/raw/pilot")
USER_COUNT = 5
MAX_USER_BYTES = 150_000_000
MAX_TOTAL_BYTES = 750_000_000

DESTINATION.mkdir(parents=True, exist_ok=True)

existing = {
    path.relative_to(DESTINATION).as_posix()
    for path in DESTINATION.rglob("*.tar")
}

print("Listing candidate user archives...", flush=True)

api = HfApi()
candidates = []

for entry in api.list_repo_tree(
    repo_id=REPO,
    repo_type="dataset",
    path_in_repo="users/0",
    recursive=False,
):
    size = getattr(entry, "size", None)

    if (
        entry.path.endswith(".tar")
        and entry.path not in existing
        and size is not None
        and 0 < size <= MAX_USER_BYTES
    ):
        candidates.append((entry.path, size))

# make selection reproducible without choosing consecutive user IDs
random.Random(42).shuffle(candidates)

selected = []
total = 0

for path, size in candidates:
    if total + size <= MAX_TOTAL_BYTES:
        selected.append((path, size))
        total += size

    if len(selected) == USER_COUNT:
        break

if not selected:
    raise RuntimeError("No additional archives fit the download limits.")

print(f"\nSelected {len(selected)} additional users:")
for path, size in selected:
    print(f"  {path}: {size / 1_000_000:.1f} MB")

print(f"\nTotal download: {total / 1_000_000:.1f} MB", flush=True)

for index, (path, size) in enumerate(selected, start=1):
    print(f"\nDownloading user {index}/{len(selected)}", flush=True)

    local_path = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=path,
        local_dir=str(DESTINATION),
    )

    print(f"Saved: {local_path}", flush=True)

print("\nBatch download complete.")
