"""Download checksum-verified public BIDSleep nights and cache epoch features.

Partial raw downloads are resumable. After a night is checksum-verified and
processed, its downloaded raw files are removed; compact features are retained.
Restarting reuses completed nights. No credentials are needed.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
import argparse
import subprocess

import pandas as pd
from preprocess_data import DEFAULT_DATASET_PATH, process_night

BASE = "https://physionet.org/files/bidsleep-dataset/1.0.0/"


def fetch_night(key, records, cache, download_cache):
    subject, night = key
    output = cache / f"{subject}_{night}.csv"
    if output.exists():
        return output
    for relative, expected in records:
        target = download_cache / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_digest(target) == expected:
            continue
        print(f"Downloading/resuming {relative}", flush=True)
        subprocess.run(["curl", "--fail", "--silent", "--show-error", "--continue-at", "-",
                        "--retry", "4", "--connect-timeout", "30", "-o", str(target),
                        BASE + relative], check=True)
        if file_digest(target) != expected:
            raise ValueError(f"Checksum mismatch: {relative}; remove the corrupt file before retrying")
    features = process_night(download_cache / subject, night)
    temporary = output.with_suffix(".tmp")
    features.to_csv(temporary, index=False)
    temporary.replace(output)
    for relative, _ in records:
        (download_cache / relative).unlink()
    return output


def file_digest(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=Path("outputs/night_features"))
    parser.add_argument("--download-cache", type=Path, default=Path("outputs/raw_downloads"))
    parser.add_argument("--output", type=Path, default=Path("processed_sleep_data_enhanced_hr.csv"))
    args = parser.parse_args()
    records = {}
    for line in (DEFAULT_DATASET_PATH / "SHA256SUMS.txt").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        parts = relative.split("/")
        if len(parts) == 3 and parts[2] in {"hr.csv", "motion.csv", "labels.mat"}:
            records.setdefault(tuple(parts[:2]), []).append((relative, digest))
    if not records or any(len(values) != 3 for values in records.values()):
        raise ValueError("Incomplete checksum manifest")
    args.cache.mkdir(parents=True, exist_ok=True)
    paths = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(fetch_night, key, value, args.cache, args.download_cache): key
                for key, value in records.items()}
        for job in as_completed(jobs):
            paths.append(job.result())
            print(f"Prepared {len(paths)}/{len(jobs)} nights: {jobs[job]}", flush=True)
    data = pd.concat([pd.read_csv(path) for path in sorted(paths)], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Saved {len(data)} epochs, {data.subject.nunique()} subjects to {args.output}")


if __name__ == "__main__":
    main()
