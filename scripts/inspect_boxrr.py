from pathlib import Path
import tarfile

from huggingface_hub import HfApi, hf_hub_download

repo_id = "cschell/boxrr-23"
output_dir = Path("data/raw/pilot")
output_dir.mkdir(parents=True, exist_ok=True)

# find one user archive without downloading a full dataset chunk
api = HfApi()
candidate = None

print("Looking for a pilot user archive...", flush=True)

for entry in api.list_repo_tree(
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="users/0",
    recursive=False,
):
    if entry.path.endswith(".tar"):
        candidate = entry.path
        break

if candidate is None:
    raise RuntimeError("No user .tar archive found under users/0.")

print(f"Selected: {candidate}", flush=True)

archive_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename=candidate,
    local_dir=str(output_dir),
)

print(f"\nDownloaded: {archive_path}")

# inspect archive members without extracting recordings
with tarfile.open(archive_path, "r:*") as archive:
    members = [member for member in archive if member.isfile()]
    recordings = [
        member for member in members
        if member.name.lower().endswith(".xror")
    ]

    print(f"\nTotal files: {len(members)}")
    print(f"XROR recordings: {len(recordings)}")

    print("\nFirst 10 files:")
    for member in members[:10]:
        print(f"  {member.name} | {member.size:,} bytes")

print("\nPilot inspection complete.")
