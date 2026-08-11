"""Generate the synthetic IAGA-2002 fixtures used by the test suite.

The fixtures are synthetic rather than real observatory data. INTERMAGNET
data are licensed CC-BY-NC, and committing them to a public repository
would be inappropriate; synthetic files also let us construct the specific
defects the validator must detect.

Run from the repository root:

    python tests/fixtures/_make_fixtures.py
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent

LABEL_WIDTH = 24
VALUE_WIDTH = 45


def header_record(label: str, value: str) -> str:
    return f"{label:<{LABEL_WIDTH}}{value:<{VALUE_WIDTH}}|"


def build(
    code: str,
    station: str,
    latitude: float,
    longitude: float,
    elevation: float,
    elements: str,
    start: datetime,
    n_samples: int,
    step: timedelta,
    baseline: dict[str, float],
    gaps: tuple[int, ...] = (),
    missing: tuple[tuple[int, str], ...] = (),
    not_observed_element: str | None = None,
) -> str:
    lines = [
        header_record("Format", "IAGA-2002"),
        header_record("Source of Data", "Synthetic test data"),
        header_record("Station Name", station),
        header_record("IAGA CODE", code),
        header_record("Geodetic Latitude", f"{latitude:.3f}"),
        header_record("Geodetic Longitude", f"{longitude:.3f}"),
        header_record("Elevation", f"{elevation:.3f}"),
        header_record("Reported", elements),
        header_record("Sensor Orientation", "HDZF"),
        header_record("Digital Sampling", "0.01 second"),
        header_record("Data Interval Type", "filtered 1-minute (00:15-01:45)"),
        header_record("Data Type", "definitive"),
        "# Synthetic data generated for unit tests. Not observatory data.",
        "# This file is not derived from any INTERMAGNET product.",
    ]

    columns = " ".join(f"{code.upper()}{e:<6}" for e in elements)
    lines.append(f"DATE       TIME         DOY     {columns}|")

    for i in range(n_samples):
        if i in gaps:
            continue
        stamp = start + i * step
        doy = stamp.timetuple().tm_yday
        values = []
        for element in elements:
            if (i, element) in missing:
                values.append(99999.00)
            elif element == not_observed_element:
                values.append(88888.00)
            else:
                wave = 8.0 * math.sin(2 * math.pi * i / 60.0)
                values.append(baseline[element] + wave)
        body = " ".join(f"{v:9.2f}" for v in values)
        lines.append(
            f"{stamp:%Y-%m-%d %H:%M:%S}.{stamp.microsecond // 1000:03d} "
            f"{doy:>3d}     {body}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    start = datetime(2024, 5, 10, 0, 0, tzinfo=UTC)
    baseline = {"X": 17200.0, "Y": -1300.0, "Z": 45900.0, "F": 49100.0}

    clean = build(
        code="esk",
        station="Eskdalemuir",
        latitude=55.314,
        longitude=356.794,
        elevation=245.0,
        elements="XYZF",
        start=start,
        n_samples=120,
        step=timedelta(minutes=1),
        baseline=baseline,
    )
    (HERE / "esk_clean_min.min").write_text(clean, encoding="utf-8")

    defective = build(
        code="esk",
        station="Eskdalemuir",
        latitude=55.314,
        longitude=356.794,
        elevation=245.0,
        elements="XYZF",
        start=start,
        n_samples=120,
        step=timedelta(minutes=1),
        baseline=baseline,
        gaps=(30, 31, 32, 90),
        missing=((10, "X"), (11, "X"), (11, "Y")),
    )
    (HERE / "esk_defective_min.min").write_text(defective, encoding="utf-8")

    partial = build(
        code="had",
        station="Hartland",
        latitude=50.995,
        longitude=355.516,
        elevation=95.0,
        elements="XYZF",
        start=start,
        n_samples=60,
        step=timedelta(minutes=1),
        baseline=baseline,
        not_observed_element="F",
    )
    (HERE / "had_partial_min.min").write_text(partial, encoding="utf-8")


if __name__ == "__main__":
    main()
