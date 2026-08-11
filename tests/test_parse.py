"""Tests for :mod:`magpipe.parse`.

The suite covers three classes of behaviour:

  * correct parsing of well-formed input (headers, index, values);
  * correct handling of the format's absent-value sentinels;
  * rejection of malformed input with a diagnostic naming the line.

All fixtures are synthetic and no test touches the network, so the
suite runs identically on a developer machine and in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from magpipe.parse import (
    MISSING_VALUE,
    NOT_OBSERVED_VALUE,
    Iaga2002Error,
    parse_file,
    parse_text,
    sentinel_counts,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def clean() -> Path:
    return FIXTURES / "esk_clean_min.min"


@pytest.fixture
def defective() -> Path:
    return FIXTURES / "esk_defective_min.min"


@pytest.fixture
def partial() -> Path:
    return FIXTURES / "had_partial_min.min"


def minimal_text(
    *,
    reported: str = "XYZF",
    columns: str = "ESKX      ESKY      ESKZ      ESKF",
    rows: str = (
        "2024-05-10 00:00:00.000 131      17200.00  -1300.00  45900.00  49100.00"
    ),
    drop_header: str | None = None,
) -> str:
    """Build a small in-memory file, optionally with one defect."""
    header = [
        ("Format", "IAGA-2002"),
        ("Source of Data", "Synthetic"),
        ("Station Name", "Eskdalemuir"),
        ("IAGA CODE", "ESK"),
        ("Geodetic Latitude", "55.314"),
        ("Geodetic Longitude", "356.794"),
        ("Elevation", "245.000"),
        ("Reported", reported),
        ("Sensor Orientation", "HDZF"),
        ("Digital Sampling", "0.01 second"),
        ("Data Interval Type", "filtered 1-minute"),
        ("Data Type", "definitive"),
    ]
    lines = [
        f"{label:<24}{value:<45}|" for label, value in header if label != drop_header
    ]
    lines.append(f"DATE       TIME         DOY     {columns}|")
    if rows:
        lines.extend(rows.splitlines())
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------
# Well-formed input
# --------------------------------------------------------------------


def test_header_records_are_parsed(clean: Path) -> None:
    result = parse_file(clean)
    assert result.header["Format"] == "IAGA-2002"
    assert result.header["Station Name"] == "Eskdalemuir"
    assert result.header["Data Type"] == "definitive"


def test_iaga_code_is_normalised_to_upper_case(clean: Path) -> None:
    # The fixture writes the code in lower case, which real files do.
    assert parse_file(clean).iaga_code == "ESK"


def test_geodetic_fields_are_numeric(clean: Path) -> None:
    result = parse_file(clean)
    assert result.latitude == pytest.approx(55.314)
    assert result.longitude == pytest.approx(356.794)
    assert result.elevation == pytest.approx(245.0)


def test_comment_records_are_captured_without_the_hash(clean: Path) -> None:
    comments = parse_file(clean).comments
    assert len(comments) == 2
    assert all(not c.startswith("#") for c in comments)
    assert "Synthetic" in comments[0]


def test_element_columns_follow_the_reported_header(clean: Path) -> None:
    assert parse_file(clean).elements == ("X", "Y", "Z", "F")


def test_index_is_utc_aware_and_ordered(clean: Path) -> None:
    data = parse_file(clean).data
    assert str(data.index.tz) == "UTC"
    assert data.index.name == "time"
    assert data.index.is_monotonic_increasing


def test_sample_count_and_cadence(clean: Path) -> None:
    data = parse_file(clean).data
    assert len(data) == 120
    assert (data.index.to_series().diff().dropna() == pd.Timedelta("1min")).all()


def test_values_are_read_correctly(clean: Path) -> None:
    data = parse_file(clean).data
    first = data.iloc[0]
    assert first["X"] == pytest.approx(17200.00)
    assert first["Y"] == pytest.approx(-1300.00)
    assert first["Z"] == pytest.approx(45900.00)
    assert first["F"] == pytest.approx(49100.00)
    assert data.dtypes.unique().tolist() == [np.dtype("float64")]


def test_hdzf_files_are_supported() -> None:
    text = minimal_text(
        reported="HDZF",
        columns="ESKH      ESKD      ESKZ      ESKF",
    )
    assert parse_text(text).elements == ("H", "D", "Z", "F")


def test_three_element_files_are_supported() -> None:
    text = minimal_text(
        reported="XYZ",
        columns="ESKX      ESKY      ESKZ",
        rows="2024-05-10 00:00:00.000 131      17200.00  -1300.00  45900.00",
    )
    result = parse_text(text)
    assert result.elements == ("X", "Y", "Z")
    assert len(result) == 1


def test_blank_lines_are_ignored() -> None:
    text = minimal_text().replace("DATE", "\n\nDATE", 1)
    assert len(parse_text(text)) == 1


def test_file_with_no_data_records_yields_an_empty_frame() -> None:
    result = parse_text(minimal_text(rows=""))
    assert len(result) == 0
    assert result.elements == ("X", "Y", "Z", "F")
    assert result.data.index.tz is not None


# --------------------------------------------------------------------
# Absent-value sentinels
# --------------------------------------------------------------------


def test_missing_sentinel_becomes_nan(defective: Path) -> None:
    data = parse_file(defective).data
    assert not (data == MISSING_VALUE).any().any()
    assert data["X"].isna().sum() == 2
    assert data["Y"].isna().sum() == 1


def test_not_observed_sentinel_becomes_nan(partial: Path) -> None:
    data = parse_file(partial).data
    assert not (data == NOT_OBSERVED_VALUE).any().any()
    assert data["F"].isna().all()
    assert data["X"].notna().all()


def test_sentinel_counts_distinguish_the_two_kinds(
    defective: Path, partial: Path
) -> None:
    counts = sentinel_counts(parse_file(defective).data)
    assert counts.loc["X", "missing"] == 2
    assert counts.loc["X", "not_observed"] == 0

    counts = sentinel_counts(parse_file(partial).data)
    assert counts.loc["F", "missing"] == 0
    assert counts.loc["F", "not_observed"] == 60


def test_sentinel_counts_rejects_a_foreign_frame() -> None:
    with pytest.raises(Iaga2002Error, match="no sentinel record"):
        sentinel_counts(pd.DataFrame({"X": [1.0]}))


def test_gaps_are_absent_rows_not_nan_rows(defective: Path) -> None:
    # The parser reports what the file contains. Detecting the four
    # omitted samples is the validator's job, not the parser's.
    data = parse_file(defective).data
    assert len(data) == 116
    steps = data.index.to_series().diff().dropna()
    # Three consecutive omitted samples leave a four-minute interval.
    assert steps.max() == pd.Timedelta("4min")
    assert (steps > pd.Timedelta("1min")).sum() == 2


# --------------------------------------------------------------------
# Malformed input
# --------------------------------------------------------------------


def test_missing_required_header_is_rejected() -> None:
    with pytest.raises(Iaga2002Error, match="Reported"):
        parse_text(minimal_text(drop_header="Reported"))


def test_the_error_lists_the_labels_that_were_present() -> None:
    with pytest.raises(Iaga2002Error, match="labels found: .*Station Name"):
        parse_text(minimal_text(drop_header="IAGA CODE"))


def test_an_optional_spec_header_may_be_absent(caplog) -> None:
    """A file missing a header we do not use is readable, with a warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        result = parse_text(minimal_text(drop_header="Digital Sampling"))
    assert len(result) == 1
    assert "Digital Sampling" in caplog.text


def test_header_labels_are_matched_case_insensitively() -> None:
    """Producers differ: 'IAGA CODE' and 'IAGA Code' both occur."""
    text = minimal_text().replace("IAGA CODE  ", "IAGA Code  ", 1)
    result = parse_text(text)
    assert result.iaga_code == "ESK"
    assert result.header["IAGA CODE"] == "ESK"


def test_extra_whitespace_in_a_label_is_tolerated() -> None:
    text = minimal_text().replace("Data Type", "Data  Type", 1)
    assert parse_text(text).header["Data Type"] == "definitive"


def test_an_unrecognised_label_is_kept_verbatim() -> None:
    extra = f"{'Comment Field':<24}{'anything':<45}|"
    text = minimal_text().replace("DATE", extra + "\nDATE", 1)
    assert parse_text(text).header["Comment Field"] == "anything"


def test_absent_data_header_is_rejected() -> None:
    text = "\n".join(
        line for line in minimal_text().splitlines() if not line.startswith("DATE")
    )
    with pytest.raises(Iaga2002Error, match="data record before the data header"):
        parse_text(text)


def test_data_header_with_wrong_leading_fields_is_rejected() -> None:
    text = minimal_text().replace(
        "DATE       TIME         DOY", "DATE       TIME         XXX", 1
    )
    with pytest.raises(Iaga2002Error, match="must begin DATE TIME DOY"):
        parse_text(text)


def test_column_count_mismatch_is_rejected() -> None:
    short = "2024-05-10 00:00:00.000 131      17200.00  -1300.00"
    with pytest.raises(Iaga2002Error, match="expected 7 fields, found 5"):
        parse_text(minimal_text(rows=short))


def test_columns_inconsistent_with_reported_header_are_rejected() -> None:
    text = minimal_text(reported="XYZF", columns="ESKH      ESKD      ESKZ      ESKF")
    with pytest.raises(Iaga2002Error, match="do not match the Reported header"):
        parse_text(text)


def test_error_message_names_the_offending_line() -> None:
    rows = (
        "2024-05-10 00:00:00.000 131      17200.00  -1300.00  45900.00  49100.00\n"
        "2024-05-10 00:01:00.000 131      17200.00  -1300.00"
    )
    with pytest.raises(Iaga2002Error, match="line 15"):
        parse_text(minimal_text(rows=rows))


def test_non_numeric_value_is_rejected() -> None:
    rows = "2024-05-10 00:00:00.000 131      17200.00  -1300.00  45900.00  ABC"
    with pytest.raises(ValueError):
        parse_text(minimal_text(rows=rows))
