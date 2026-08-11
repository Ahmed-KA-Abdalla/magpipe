"""Cut a small, attributed extract from a cached INTERMAGNET file.

The cache under ``data/raw/`` holds whole days at one-minute cadence,
which is around 200 kB per observatory per day and grows quickly. Those
files stay out of version control. This script produces the small
extract that the repository does commit, so that the integration tests
run against real observatory data rather than only against synthetic
fixtures.

The extract is written to ``tests/data/`` with the INTERMAGNET
acknowledgement inserted as comment records, which is what the licence
requires of a redistributed copy.

Example
-------
::

    python scripts/make_sample.py data/raw/esk_20240510T0000_...min \
        --hours 6 --label esk_storm

A six-hour extract at one-minute cadence is 360 records, roughly 30 kB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magpipe.parse import parse_file  # noqa: E402

ACKNOWLEDGEMENT = (
    "The data in this file are derived from INTERMAGNET observatory data,",
    "made available under the Creative Commons Attribution-NonCommercial",
    "4.0 licence (CC BY-NC 4.0). The results presented rely on data",
    "collected at magnetic observatories. We thank the national institutes",
    "that support them and INTERMAGNET for promoting high standards of",
    "magnetic observatory practice (https://intermagnet.org).",
    "This is a truncated extract retained for automated testing only.",
)


def extract(path: Path, hours: float, out: Path) -> Path:
    """Write the first ``hours`` of ``path`` to ``out``, with attribution."""
    parsed = parse_file(path)
    if parsed.data.empty:
        raise SystemExit(f"{path} contains no data records")

    cutoff = parsed.data.index.min() + pd.Timedelta(hours=hours)
    keep = parsed.data.index < cutoff
    n_keep = int(keep.sum())
    if n_keep == 0:
        raise SystemExit("the requested interval selects no samples")

    lines = path.read_text(encoding="utf-8").splitlines()
    header: list[str] = []
    data_header: str | None = None
    data: list[str] = []

    for line in lines:
        if not line.strip():
            continue
        if line.lstrip().upper().startswith("DATE"):
            data_header = line
        elif data_header is None:
            if not line.lstrip().startswith("#"):
                header.append(line)
        else:
            data.append(line)

    if data_header is None:
        raise SystemExit(f"{path} has no data header record")

    out.parent.mkdir(parents=True, exist_ok=True)
    body = (
        header
        + [f"# {line}" for line in ACKNOWLEDGEMENT]
        + [f"# Source file: {path.name}"]
        + [data_header]
        + data[:n_keep]
    )
    out.write_text("\n".join(body) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="a file under data/raw/")
    parser.add_argument("--hours", type=float, default=6.0, help="how much to keep")
    parser.add_argument(
        "--label",
        required=True,
        help="output name stem, e.g. esk_storm -> tests/data/esk_storm.min",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("tests/data"))
    args = parser.parse_args(argv)

    if not args.source.exists():
        raise SystemExit(f"no such file: {args.source}")

    out = extract(args.source, args.hours, args.out_dir / f"{args.label}.min")
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out} ({size_kb:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
