#!/usr/bin/env python
"""Frozen-artifact integrity check (Stage 7 of the research-extension run order). Hashes every file
under the frozen result/report/config directories and compares against a snapshot taken before the
extension work started. Never touches or reads the content of files for any purpose other than hashing.

    python scripts/verify_frozen_integrity.py --snapshot /path/to/before_hashes.txt

The snapshot file is `sha256sum`-format: one "<hash>  <relative path>" line per file, generated with e.g.
    find results report configs -type f -not -path 'results/research_extension/*' | sort | xargs sha256sum
"""
import argparse
import hashlib
import sys
from pathlib import Path

FROZEN_ROOTS = ["configs", "report/report.md", "report/step6_streaming_analysis.md",
               "results/checkpoints", "results/figures", "results/metrics", "results/experiment_log.csv",
               "results/phase1_console.txt", "results/smoke_test"]
# Note: report/report.md and report/step6_streaming_analysis.md are frozen; report/research_extension.md
# is NEW (the extension's own report) and is intentionally excluded from this list.


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def collect_current_hashes(root: Path) -> dict:
    hashes = {}
    for rel in FROZEN_ROOTS:
        p = root / rel
        if p.is_file():
            hashes[rel] = sha256_of(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    hashes[f.relative_to(root).as_posix()] = sha256_of(f)
    return hashes


def parse_snapshot(path: Path) -> dict:
    hashes = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        h, rel = line.split(None, 1)
        hashes[rel.strip()] = h
    return hashes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default=None, help="sha256sum-format hash file from BEFORE the extension ran")
    ap.add_argument("--write_snapshot", default=None,
                    help="write a snapshot of the CURRENT frozen files to this path and exit (run BEFORE training)")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()

    root = Path(args.root)
    if args.write_snapshot:
        cur = collect_current_hashes(root)
        Path(args.write_snapshot).write_text("".join(f"{h}  {rel}\n" for rel, h in sorted(cur.items())))
        print(f"Wrote snapshot of {len(cur)} frozen files to {args.write_snapshot}")
        return 0
    if not args.snapshot:
        ap.error("--snapshot is required unless --write_snapshot is given")
    before = parse_snapshot(Path(args.snapshot))
    after = collect_current_hashes(root)

    changed, missing, extra = [], [], []
    for rel, h in before.items():
        if rel not in after:
            missing.append(rel)
        elif after[rel] != h:
            changed.append(rel)
    for rel in after:
        if rel not in before:
            extra.append(rel)

    print(f"Checked {len(before)} frozen files under {FROZEN_ROOTS}.")
    if changed:
        print(f"\n[FAIL] {len(changed)} frozen file(s) CHANGED:")
        for r in changed:
            print(f"  - {r}")
    if missing:
        print(f"\n[FAIL] {len(missing)} frozen file(s) MISSING:")
        for r in missing:
            print(f"  - {r}")
    if extra:
        print(f"\n[NOTE] {len(extra)} file(s) present now but not in the snapshot "
             f"(new files created under a frozen root would be unusual -- review):")
        for r in extra:
            print(f"  - {r}")

    ok = not changed and not missing
    print(f"\n{'[PASS]' if ok else '[FAIL]'} Frozen-artifact integrity check {'passed' if ok else 'FAILED'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
