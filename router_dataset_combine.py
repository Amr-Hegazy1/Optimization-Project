from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


KEYS = ("sa", "ga", "aco")


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise SystemExit(f"Input has no header: {path}")
        return list(reader), fieldnames


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _counts(rows: Iterable[dict[str, str]]) -> Dict[str, int]:
    c: Dict[str, int] = defaultdict(int)
    for r in rows:
        c[str(r.get("label", ""))] += 1
    return c


def _cap_per_class(rows: list[dict[str, str]], *, cap: int, seed: int) -> list[dict[str, str]]:
    if cap <= 0:
        return rows
    rng = random.Random(int(seed))
    groups: Dict[str, list[dict[str, str]]] = defaultdict(list)
    other: list[dict[str, str]] = []
    for r in rows:
        lab = str(r.get("label", ""))
        if lab in KEYS:
            groups[lab].append(r)
        else:
            other.append(r)

    out: list[dict[str, str]] = []
    for k in KEYS:
        g = groups.get(k, [])
        if len(g) > cap:
            g = rng.sample(g, k=cap)
        out.extend(g)
    out.extend(other)
    rng.shuffle(out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Combine router datasets (CSV) and optionally cap samples per class.")
    p.add_argument("--in", dest="inputs", action="append", required=True, help="Input CSV path. Repeat for multiple.")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument(
        "--cap-per-class",
        type=int,
        default=0,
        help="If set, downsample each of sa/ga/aco to at most this many rows after combining.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    input_paths = [Path(x) for x in args.inputs]
    for ip in input_paths:
        if not ip.exists():
            raise SystemExit(f"Input not found: {ip}")

    all_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None

    for ip in input_paths:
        rows, fns = _read_rows(ip)
        if fieldnames is None:
            fieldnames = fns
        else:
            # Require identical schema for safety.
            missing = [x for x in fieldnames if x not in fns]
            extra = [x for x in fns if x not in fieldnames]
            if missing or extra:
                raise SystemExit(
                    f"Schema mismatch for {ip}. Missing={missing} Extra={extra}"
                )
        all_rows.extend(rows)

    assert fieldnames is not None

    rng = random.Random(int(args.seed))
    rng.shuffle(all_rows)

    before = _counts(all_rows)
    all_rows = _cap_per_class(all_rows, cap=int(args.cap_per_class), seed=int(args.seed))
    after = _counts(all_rows)

    out = Path(args.out)
    _write_rows(out, all_rows, fieldnames)

    print(f"Wrote {len(all_rows)} rows to {out}")
    print("Counts before:", {k: before.get(k, 0) for k in KEYS})
    print("Counts after :", {k: after.get(k, 0) for k in KEYS})


if __name__ == "__main__":
    main()
