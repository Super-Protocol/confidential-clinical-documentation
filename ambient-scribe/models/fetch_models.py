#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Super Protocol
# SPDX-License-Identifier: Apache-2.0
#
# Added by Super Protocol, 2026.
"""Fetch the model files named in `manifest.json` and refuse any byte that is not pinned.

Run at IMAGE BUILD time, never at runtime. Everything this application needs to
recognise speech and separate speakers ends up inside the image, which means:

* a deployment reaches no model host — not huggingface.co, not NGC, nothing — so
  a confidential cluster space with no egress at all still works;
* the image digest the listing pins covers the weights as well as the code, so
  "which model touched this consultation" is answered by the attestation rather
  than by a promise.

A file whose sha256 does not match the manifest is a hard failure. The size is
checked too, because a proxy serving an HTML error page is the usual cause and it
is clearer to say "4 KB, expected 1.5 GB" than to print two hashes.

Usage:
    python3 fetch_models.py [--manifest manifest.json] [--root /]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "confidential-clinical-documentation/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, CHUNK)
    tmp.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("manifest.json")))
    parser.add_argument("--root", default="/", help="prefix for each group's target directory")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text("utf-8"))
    root = Path(args.root)
    failures = 0
    fetched = 0

    for group, spec in manifest.items():
        if group.startswith("_"):
            continue
        target_dir = root / spec["target"].lstrip("/")
        for entry in spec["files"]:
            name = entry.get("rename", entry["path"])
            target = target_dir / name
            url = HF.format(repo=spec["repo"], revision=spec["revision"], path=entry["path"])

            if not target.exists():
                print(f"fetch  {group}/{name}", flush=True)
                fetch(url, target)
                fetched += 1

            size = target.stat().st_size
            if size != entry["size"]:
                print(f"FAIL   {group}/{name}: {size} bytes, the manifest says {entry['size']}",
                      file=sys.stderr)
                failures += 1
                continue
            digest = sha256_of(target)
            if digest != entry["sha256"]:
                print(f"FAIL   {group}/{name}: sha256 {digest}, "
                      f"the manifest says {entry['sha256']}", file=sys.stderr)
                failures += 1
                continue
            print(f"ok     {group}/{name}  {size} bytes  {digest[:16]}…", flush=True)

    # The licences the weights are served under, beside the weights, in the image.
    licences = root / "licenses/models"
    licences.mkdir(parents=True, exist_ok=True)
    (licences / "MODEL-LICENCES.json").write_text(
        json.dumps({g: s["licence"] | {"title": s["title"], "repo": s["repo"],
                                       "revision": s["revision"]}
                    for g, s in manifest.items() if not g.startswith("_")}, indent=2) + "\n",
        encoding="utf-8")
    seg_licence = root / manifest["diarization_segmentation"]["target"].lstrip("/") / "LICENSE.segmentation"
    if seg_licence.exists():
        shutil.copy2(seg_licence, licences / "LICENSE.pyannote-segmentation-3.0")

    print(f"\n{fetched} file(s) fetched, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
