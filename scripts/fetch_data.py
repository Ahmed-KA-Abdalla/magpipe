"""Download observatory data from the INTERMAGNET GIN into the local cache.

Examples
--------
One day from two UK observatories over the May 2024 storm::

    python scripts/fetch_data.py --obs ESK HAD --start 2024-05-10 --check

Three days, with the total field taken from the independent scalar
instrument so that the vector/scalar consistency check is meaningful::

    python scripts/fetch_data.py --obs ESK --start 2024-05-10 --days 3 \
        --orientation XYZS --check

Files land in ``data/raw/``, which is excluded from version control.
Re-running is cheap: a cached file is not downloaded again unless
``--refresh`` is given.

INTERMAGNET data are CC-BY-NC and subject to INTERMAGNET's conditions of
use. See the acknowledgement in README.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magpipe.fetch import (  # noqa: E402
    ORIENTATIONS,
    PUBLICATION_STATES,
    DataRequest,
    fetch_many,
)
from magpipe.parse import Iaga2002Error, parse_file  # noqa: E402
from magpipe.validate import validate  # noqa: E402


def parse_day(text: str) -> date:
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a date as YYYY-MM-DD, got {text!r}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--obs",
        nargs="+",
        required=True,
        metavar="CODE",
        help="three-letter IAGA observatory codes, e.g. ESK HAD",
    )
    parser.add_argument("--start", type=parse_day, required=True)
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="number of days from --start (default 1)",
    )
    parser.add_argument("--cadence", default="Minute", choices=["Minute", "Second"])
    parser.add_argument(
        "--publication-state", default="best-avail", choices=list(PUBLICATION_STATES)
    )
    parser.add_argument(
        "--orientation",
        default="Native",
        choices=list(ORIENTATIONS),
        help=(
            "XYZS or HDZS take F from an independent scalar instrument, "
            "which makes the vector/scalar consistency check meaningful; "
            "XYZF and HDZF compute F from the vector components, which "
            "makes that check a tautology"
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download again even if a cached copy exists",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="parse and validate each file after download",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    try:
        requests_ = [
            DataRequest(
                observatory=code,
                start=args.start,
                days=args.days,
                cadence=args.cadence,
                publication_state=args.publication_state,
                orientation=args.orientation,
            )
            for code in args.obs
        ]
    except ValueError as exc:
        parser.error(str(exc))

    with requests.Session() as session:
        results = fetch_many(
            requests_,
            session,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
        )

    if not results:
        print("\nNo files retrieved.", file=sys.stderr)
        return 1

    failed = len(requests_) - len(results)
    print(f"\nRetrieved {len(results)} of {len(requests_)} file(s).")

    if args.check:
        for request, path in results.items():
            print(f"\n{request.code}  {path.name}")
            try:
                parsed = parse_file(path)
            except Iaga2002Error as exc:
                print(f"  PARSE FAILED: {exc}")
                failed += 1
                continue
            if not len(parsed):
                print("  file contains no data records")
                failed += 1
                continue
            report = validate(parsed.data)
            print(
                f"  {len(parsed)} samples, elements "
                f"{''.join(parsed.elements)}, "
                f"{parsed.data.index.min()} to {parsed.data.index.max()}"
            )
            print(f"  data type: {parsed.header.get('Data Type', 'unknown')}")
            for line in report.summary().splitlines():
                print(f"  {line}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
