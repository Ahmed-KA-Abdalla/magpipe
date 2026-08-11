"""Parser for the IAGA-2002 geomagnetic data exchange format.

The format is specified in the INTERMAGNET Technical Reference Manual,
Appendix E3. A file consists of:

  * twelve mandatory header records, each 69 characters wide with a
    label in columns 1-24, a value in columns 25-69 and a terminating
    ``|`` in column 70;
  * zero or more optional comment records, identified by ``#`` in
    column 1;
  * one data header record beginning ``DATE``, naming the columns;
  * one data record per sample.

Absent values are encoded with numeric sentinels rather than blanks:
99999.00 for a missing value and 88888.00 for a value that was not
observed. Both are converted to NaN here, but the distinction is
preserved in :func:`sentinel_counts` because it matters for data
quality reporting -- a missing value indicates a fault, whereas a
value that was never observed indicates an instrument that does not
report that component.

This module performs no network access and no database access, so it
can be tested in isolation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "Iaga2002Error",
    "SPEC_HEADERS",
    "REQUIRED_HEADERS",
    "MISSING_VALUE",
    "NOT_OBSERVED_VALUE",
    "MagnetogramFile",
    "parse_text",
    "parse_file",
    "sentinel_counts",
]

logger = logging.getLogger(__name__)

MISSING_VALUE = 99999.0
NOT_OBSERVED_VALUE = 88888.0

#: The twelve header labels the specification mandates. Labels are
#: matched case-insensitively and with runs of whitespace collapsed,
#: because producers differ on capitalisation -- "IAGA CODE" and "IAGA
#: Code" both occur in files served by INTERMAGNET. Whichever spelling
#: arrives, it is stored under the canonical form below so that callers
#: have one name to use.
SPEC_HEADERS = (
    "Format",
    "Source of Data",
    "Station Name",
    "IAGA CODE",
    "Geodetic Latitude",
    "Geodetic Longitude",
    "Elevation",
    "Reported",
    "Sensor Orientation",
    "Digital Sampling",
    "Data Interval Type",
    "Data Type",
)

#: The subset this module cannot work without. The remaining spec
#: headers are reported as a warning when absent rather than an error:
#: a file that is missing "Digital Sampling" is still perfectly
#: readable, and refusing it would reject usable data for no benefit.
REQUIRED_HEADERS = ("Format", "IAGA CODE", "Reported")

_CANONICAL = {" ".join(h.split()).lower(): h for h in SPEC_HEADERS}

_LABEL_WIDTH = 24
_RECORD_WIDTH = 69


class Iaga2002Error(ValueError):
    """Raised when input does not conform to the IAGA-2002 format."""


@dataclass(frozen=True)
class MagnetogramFile:
    """A parsed IAGA-2002 file.

    Attributes
    ----------
    header:
        The twelve mandatory header records, keyed by label.
    comments:
        Optional comment records, with the leading ``#`` stripped.
    data:
        One row per sample, indexed by a UTC-aware timestamp. Columns
        are the single-letter element codes given in the ``Reported``
        header, for example ``X``, ``Y``, ``Z``, ``F``.
    """

    header: Mapping[str, str]
    comments: tuple[str, ...]
    data: pd.DataFrame

    @property
    def iaga_code(self) -> str:
        return self.header["IAGA CODE"].upper()

    @property
    def elements(self) -> tuple[str, ...]:
        return tuple(self.data.columns)

    @property
    def latitude(self) -> float:
        return float(self.header["Geodetic Latitude"])

    @property
    def longitude(self) -> float:
        return float(self.header["Geodetic Longitude"])

    @property
    def elevation(self) -> float:
        return float(self.header["Elevation"])

    def __len__(self) -> int:
        return len(self.data)


def parse_file(path: str | Path, encoding: str = "utf-8") -> MagnetogramFile:
    """Parse an IAGA-2002 file from disk."""
    text = Path(path).read_text(encoding=encoding)
    return parse_text(text)


def parse_text(text: str) -> MagnetogramFile:
    """Parse IAGA-2002 content held in a string."""
    header: dict[str, str] = {}
    comments: list[str] = []
    column_names: list[str] | None = None
    rows: list[tuple[str, str, list[str]]] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\n\r")
        if not line.strip():
            continue

        if line.lstrip().startswith("#"):
            comments.append(line.lstrip().lstrip("#").strip())
        elif line.lstrip().upper().startswith("DATE"):
            if column_names is not None:
                raise Iaga2002Error(f"line {lineno}: second data header record found")
            column_names = _parse_data_header(line, lineno)
        elif line.rstrip().endswith("|"):
            label, value = _parse_header_record(line, lineno)
            header[label] = value
        else:
            if column_names is None:
                raise Iaga2002Error(
                    f"line {lineno}: data record before the data header record"
                )
            rows.append(_split_data_record(line, lineno, len(column_names)))

    _check_required_headers(header)
    if column_names is None:
        raise Iaga2002Error("no data header record (a line beginning DATE)")

    reported = _reported_elements(header["Reported"])
    if tuple(column_names) != reported:
        raise Iaga2002Error(
            "data header columns "
            f"{column_names} do not match the Reported header {list(reported)}"
        )

    data = _build_frame(rows, reported)
    return MagnetogramFile(header=dict(header), comments=tuple(comments), data=data)


def sentinel_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Count absent values per column, split by sentinel kind.

    Call this on the raw frame returned by :func:`parse_text`, which
    records which sentinel produced each NaN in ``frame.attrs``.
    """
    kinds = frame.attrs.get("sentinels")
    if kinds is None:
        raise Iaga2002Error(
            "frame carries no sentinel record; it was not produced by parse_text"
        )
    return pd.DataFrame(kinds, index=["missing", "not_observed"]).T


def _canonical_label(label: str) -> str:
    """Map a header label to its canonical spelling, if it is one we know."""
    return _CANONICAL.get(" ".join(label.split()).lower(), label.strip())


def _parse_header_record(line: str, lineno: int) -> tuple[str, str]:
    body = line.rstrip()[:-1] if line.rstrip().endswith("|") else line
    if len(body) >= _LABEL_WIDTH:
        label = body[:_LABEL_WIDTH].strip()
        value = body[_LABEL_WIDTH:].strip()
    else:
        parts = body.split(None, 1)
        label = parts[0].strip() if parts else ""
        value = parts[1].strip() if len(parts) > 1 else ""
    if not label:
        raise Iaga2002Error(f"line {lineno}: header record has no label")
    return _canonical_label(label), value


def _parse_data_header(line: str, lineno: int) -> list[str]:
    body = line.rstrip()
    if body.endswith("|"):
        body = body[:-1]
    fields = body.split()
    if len(fields) < 4:
        raise Iaga2002Error(f"line {lineno}: data header record names no data columns")
    if [f.upper() for f in fields[:3]] != ["DATE", "TIME", "DOY"]:
        raise Iaga2002Error(
            f"line {lineno}: data header must begin DATE TIME DOY, found {fields[:3]}"
        )
    # Column names are the IAGA code followed by the element letter,
    # for example ESKX. Take the final character.
    return [f[-1].upper() for f in fields[3:]]


def _split_data_record(
    line: str, lineno: int, n_columns: int
) -> tuple[str, str, list[str]]:
    fields = line.split()
    expected = 3 + n_columns
    if len(fields) != expected:
        raise Iaga2002Error(
            f"line {lineno}: expected {expected} fields, found {len(fields)}"
        )
    return fields[0], fields[1], fields[3:]


def _reported_elements(reported: str) -> tuple[str, ...]:
    letters = tuple(ch.upper() for ch in reported.strip() if ch.isalpha())
    if not letters:
        raise Iaga2002Error("the Reported header names no elements")
    return letters


def _check_required_headers(header: Mapping[str, str]) -> None:
    """Fail on headers this module needs; warn about the rest.

    The error names the labels that were actually present, because the
    usual cause of a failure here is a producer spelling a label
    differently rather than omitting it, and that is impossible to
    diagnose from the absence alone.
    """
    missing = [key for key in REQUIRED_HEADERS if key not in header]
    if missing:
        found = ", ".join(sorted(header)) or "none"
        raise Iaga2002Error(
            "missing required header record(s): "
            + ", ".join(missing)
            + f"; labels found: {found}"
        )

    absent = [key for key in SPEC_HEADERS if key not in header]
    if absent:
        logger.warning(
            "file omits header record(s) mandated by the specification: %s",
            ", ".join(absent),
        )


def _build_frame(
    rows: Iterable[tuple[str, str, list[str]]], elements: tuple[str, ...]
) -> pd.DataFrame:
    rows = list(rows)
    if not rows:
        empty = pd.DataFrame(
            {element: pd.Series(dtype="float64") for element in elements},
            index=pd.DatetimeIndex([], tz="UTC", name="time"),
        )
        empty.attrs["sentinels"] = {e: (0, 0) for e in elements}
        return empty

    stamps = [f"{date} {time}" for date, time, _ in rows]
    index = pd.to_datetime(stamps, format="ISO8601", utc=True)
    index.name = "time"

    values = np.array([values for _, _, values in rows], dtype="float64")
    frame = pd.DataFrame(values, columns=list(elements), index=index)

    sentinels: dict[str, tuple[int, int]] = {}
    for element in elements:
        column = frame[element]
        n_missing = int((column == MISSING_VALUE).sum())
        n_absent = int((column == NOT_OBSERVED_VALUE).sum())
        sentinels[element] = (n_missing, n_absent)
    frame = frame.replace({MISSING_VALUE: np.nan, NOT_OBSERVED_VALUE: np.nan})
    frame.attrs["sentinels"] = sentinels
    return frame
