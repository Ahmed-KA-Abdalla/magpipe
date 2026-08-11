"""Produce figures from cached observatory data.

Examples
--------
Every panel for one observatory, plus the scalar/vector residual::

    python scripts/plot_day.py --obs ESK --start 2024-05-10

Two observatories overlaid on one element, to see whether a disturbance
is global::

    python scripts/plot_day.py --obs ESK HAD --start 2024-05-10 \
        --compare X

Figures are written to ``docs/figures/``. Data must already be in the
cache; run ``scripts/fetch_data.py`` first.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from magpipe.fetch import DataRequest, cache_path  # noqa: E402
from magpipe.parse import parse_file  # noqa: E402
from magpipe.plot import (  # noqa: E402
    plot_comparison,
    plot_residual,
    plot_timeseries,
    save,
)
from magpipe.validate import validate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--obs", nargs="+", required=True, metavar="CODE")
    parser.add_argument(
        "--start",
        required=True,
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
    )
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--cadence", default="Minute")
    parser.add_argument("--orientation", default="XYZS")
    parser.add_argument("--publication-state", default="best-avail")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--compare",
        metavar="ELEMENT",
        help="also overlay this element across all observatories",
    )
    args = parser.parse_args(argv)

    frames = {}
    for code in args.obs:
        request = DataRequest(
            observatory=code,
            start=args.start,
            days=args.days,
            cadence=args.cadence,
            publication_state=args.publication_state,
            orientation=args.orientation,
        )
        path = cache_path(request, args.cache_dir)
        if not path.exists():
            print(
                f"not in cache: {path.name}\n"
                f"  run scripts/fetch_data.py with the same arguments first",
                file=sys.stderr,
            )
            return 1
        parsed = parse_file(path)
        frames[request.code] = parsed.data

        span = f"{args.start}" + (f" +{args.days - 1}d" if args.days > 1 else "")
        title = f"{request.code}  {span}  ({parsed.header.get('Data Type', '')})"

        figure = plot_timeseries(parsed.data, title=title)
        out = args.out_dir / f"{request.code.lower()}_{args.start}_series.png"
        save(figure, out)
        print(f"wrote {out}")

        if {"X", "Y", "Z", "F"}.issubset(parsed.data.columns):
            figure = plot_residual(parsed.data, title=f"{title} — residual")
            out = args.out_dir / f"{request.code.lower()}_{args.start}_residual.png"
            save(figure, out)
            print(f"wrote {out}")

        report = validate(parsed.data)
        print(f"  {report.summary().splitlines()[0]}")

    if args.compare:
        figure = plot_comparison(
            frames,
            element=args.compare,
            title=f"{args.compare} anomaly, {args.start}",
        )
        out = args.out_dir / f"compare_{args.compare}_{args.start}.png"
        save(figure, out)
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
