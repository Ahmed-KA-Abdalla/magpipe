"""Tests for :mod:`magpipe.validate`.

Each check is tested twice: once against data it should pass, and once
against data carrying exactly the defect it exists to detect. Frames are
built in memory by :func:`make_frame` so that a test's intent is visible
in the test itself rather than hidden in a fixture file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from magpipe.parse import parse_file
from magpipe.validate import (
    Issue,
    Report,
    Thresholds,
    check_cadence,
    check_completeness,
    check_field_consistency,
    check_ordering,
    check_range,
    check_spikes,
    hdz_to_xyz,
    validate,
)

FIXTURES = Path(__file__).parent / "fixtures"


def make_frame(
    n: int = 60,
    start: str = "2024-05-10T00:00:00Z",
    freq: str = "1min",
    x: float = 17_200.0,
    y: float = -1_300.0,
    z: float = 45_900.0,
) -> pd.DataFrame:
    """A consistent XYZF frame: F is the exact vector magnitude."""
    index = pd.date_range(start=start, periods=n, freq=freq, tz="UTC", name="time")
    frame = pd.DataFrame(
        {
            "X": np.full(n, x),
            "Y": np.full(n, y),
            "Z": np.full(n, z),
        },
        index=index,
    )
    frame["F"] = np.sqrt(frame["X"] ** 2 + frame["Y"] ** 2 + frame["Z"] ** 2)
    return frame


@pytest.fixture
def thresholds() -> Thresholds:
    return Thresholds()


# --------------------------------------------------------------------
# A clean frame raises nothing
# --------------------------------------------------------------------


def test_clean_frame_produces_no_issues(thresholds: Thresholds) -> None:
    report = validate(make_frame(), thresholds)
    assert report.ok
    assert report.issues == ()
    assert report.summary() == "No issues."


def test_report_to_frame_is_empty_but_typed() -> None:
    table = Report().to_frame()
    assert table.empty
    assert list(table.columns) == [
        "check",
        "severity",
        "element",
        "start",
        "end",
        "count",
        "message",
    ]


# --------------------------------------------------------------------
# Cadence
# --------------------------------------------------------------------


def test_cadence_accepts_a_regular_series(thresholds: Thresholds) -> None:
    assert check_cadence(make_frame(), thresholds) == []


def test_cadence_reports_a_single_absent_sample(thresholds: Thresholds) -> None:
    frame = make_frame(n=10).drop(index=pd.Timestamp("2024-05-10T00:04:00Z"))
    issues = check_cadence(frame, thresholds)
    assert len(issues) == 1
    assert issues[0].count == 1
    assert issues[0].severity == "error"
    assert issues[0].start == pd.Timestamp("2024-05-10T00:04:00Z")
    assert issues[0].end == pd.Timestamp("2024-05-10T00:04:00Z")


def test_cadence_reports_a_run_as_one_interval(thresholds: Thresholds) -> None:
    absent = pd.to_datetime(
        ["2024-05-10T00:03:00Z", "2024-05-10T00:04:00Z", "2024-05-10T00:05:00Z"]
    )
    frame = make_frame(n=12).drop(index=absent)
    issues = check_cadence(frame, thresholds)
    assert len(issues) == 1
    assert issues[0].count == 3
    assert issues[0].start == pd.Timestamp("2024-05-10T00:03:00Z")
    assert issues[0].end == pd.Timestamp("2024-05-10T00:05:00Z")


def test_cadence_reports_two_separate_gaps(thresholds: Thresholds) -> None:
    frame = make_frame(n=20).drop(
        index=pd.to_datetime(["2024-05-10T00:04:00Z", "2024-05-10T00:12:00Z"])
    )
    assert len(check_cadence(frame, thresholds)) == 2


def test_cadence_flags_an_interval_that_is_not_a_multiple() -> None:
    index = pd.to_datetime(
        [
            "2024-05-10T00:00:00Z",
            "2024-05-10T00:01:00Z",
            "2024-05-10T00:01:30Z",
        ]
    )
    frame = pd.DataFrame({"X": [1.0, 2.0, 3.0]}, index=index)
    issues = check_cadence(frame, Thresholds())
    assert any("whole multiple" in i.message for i in issues)


def test_cadence_respects_a_different_expected_interval() -> None:
    frame = make_frame(n=10, freq="1s")
    assert check_cadence(frame, Thresholds(expected_interval=pd.Timedelta("1s"))) == []
    assert check_cadence(frame, Thresholds()) != []


# --------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------


def test_ordering_accepts_a_sorted_unique_index(thresholds: Thresholds) -> None:
    assert check_ordering(make_frame(), thresholds) == []


def test_ordering_detects_duplicates(thresholds: Thresholds) -> None:
    frame = make_frame(n=5)
    frame = pd.concat([frame, frame.iloc[[2]]]).sort_index()
    issues = check_ordering(frame, thresholds)
    assert any("duplicated" in i.message for i in issues)


def test_ordering_detects_reversed_samples(thresholds: Thresholds) -> None:
    frame = make_frame(n=5).iloc[::-1]
    issues = check_ordering(frame, thresholds)
    assert any("increasing order" in i.message for i in issues)


# --------------------------------------------------------------------
# Completeness
# --------------------------------------------------------------------


def test_completeness_accepts_a_full_frame(thresholds: Thresholds) -> None:
    assert check_completeness(make_frame(), thresholds) == []


def test_completeness_warns_below_the_tolerance(thresholds: Thresholds) -> None:
    frame = make_frame(n=100)
    frame.iloc[3, frame.columns.get_loc("X")] = np.nan
    issues = check_completeness(frame, thresholds)
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert issues[0].element == "X"


def test_completeness_errors_above_the_tolerance(thresholds: Thresholds) -> None:
    frame = make_frame(n=100)
    frame.iloc[:20, frame.columns.get_loc("X")] = np.nan
    issues = check_completeness(frame, thresholds)
    assert issues[0].severity == "error"
    assert issues[0].count == 20


def test_completeness_rejects_an_empty_frame(thresholds: Thresholds) -> None:
    issues = check_completeness(make_frame(n=0), thresholds)
    assert issues[0].severity == "error"
    assert "no samples" in issues[0].message


# --------------------------------------------------------------------
# Range
# --------------------------------------------------------------------


def test_range_accepts_plausible_values(thresholds: Thresholds) -> None:
    assert check_range(make_frame(), thresholds) == []


def test_range_rejects_an_implausible_component(thresholds: Thresholds) -> None:
    frame = make_frame(n=10)
    frame.iloc[5, frame.columns.get_loc("X")] = 250_000.0
    issues = check_range(frame, thresholds)
    assert [i.element for i in issues] == ["X"]
    assert issues[0].severity == "error"


def test_range_rejects_a_total_field_below_the_floor(
    thresholds: Thresholds,
) -> None:
    frame = make_frame(n=10)
    frame.iloc[2, frame.columns.get_loc("F")] = 100.0
    issues = check_range(frame, thresholds)
    assert any(i.element == "F" for i in issues)


def test_range_ignores_absent_values(thresholds: Thresholds) -> None:
    frame = make_frame(n=10)
    frame.iloc[2, frame.columns.get_loc("X")] = np.nan
    assert check_range(frame, thresholds) == []


# --------------------------------------------------------------------
# Spikes
# --------------------------------------------------------------------


def test_spikes_accepts_a_smooth_series(thresholds: Thresholds) -> None:
    assert check_spikes(make_frame(), thresholds) == []


def test_spikes_detects_a_single_discontinuity(thresholds: Thresholds) -> None:
    frame = make_frame(n=10)
    frame.iloc[5, frame.columns.get_loc("X")] += 2_000.0
    issues = check_spikes(frame, thresholds)
    # A single displaced sample produces a step in and a step out.
    assert issues[0].element == "X"
    assert issues[0].count == 2
    assert issues[0].severity == "warning"


def test_spikes_tolerates_storm_scale_variation(thresholds: Thresholds) -> None:
    frame = make_frame(n=60)
    frame["X"] = frame["X"] + np.linspace(0, 1_200, 60)  # 20 nT per minute
    assert check_spikes(frame, thresholds) == []


# --------------------------------------------------------------------
# Field consistency
# --------------------------------------------------------------------


def test_field_consistency_accepts_a_consistent_frame(
    thresholds: Thresholds,
) -> None:
    assert check_field_consistency(make_frame(), thresholds) == []


def test_field_consistency_detects_a_systematic_offset(
    thresholds: Thresholds,
) -> None:
    frame = make_frame(n=10)
    frame["F"] = frame["F"] + 25.0
    issues = check_field_consistency(frame, thresholds)
    errors = [i for i in issues if i.severity == "error"]
    assert len(errors) == 1
    assert "systematic offset" in errors[0].message
    assert "+25.00" in errors[0].message


def test_field_consistency_tolerates_ordinary_instrument_scatter(
    thresholds: Thresholds,
) -> None:
    """Scatter of the size measured at ESK and HAD must not be flagged."""
    rng = np.random.default_rng(0)
    frame = make_frame(n=1440)
    frame["F"] = frame["F"] + rng.normal(0.13, 0.52, len(frame))
    assert check_field_consistency(frame, thresholds) == []


def test_field_consistency_tolerates_a_few_large_excursions(
    thresholds: Thresholds,
) -> None:
    frame = make_frame(n=1440)
    frame.iloc[:5, frame.columns.get_loc("F")] += 8.0
    assert check_field_consistency(frame, thresholds) == []


def test_field_consistency_warns_when_excursions_are_common(
    thresholds: Thresholds,
) -> None:
    frame = make_frame(n=1000)
    frame.iloc[:50, frame.columns.get_loc("F")] += 8.0
    issues = check_field_consistency(frame, thresholds)
    warnings = [i for i in issues if i.severity == "warning"]
    assert len(warnings) == 1
    assert warnings[0].count == 50


def test_field_consistency_is_skipped_without_all_four_elements(
    thresholds: Thresholds,
) -> None:
    frame = make_frame(n=10).drop(columns=["F"])
    assert check_field_consistency(frame, thresholds) == []


def test_field_consistency_ignores_rows_with_absent_values(
    thresholds: Thresholds,
) -> None:
    frame = make_frame(n=10)
    frame.iloc[3, frame.columns.get_loc("F")] = np.nan
    assert check_field_consistency(frame, thresholds) == []


# --------------------------------------------------------------------
# HDZ conversion
# --------------------------------------------------------------------


def test_hdz_to_xyz_recovers_the_components() -> None:
    index = pd.date_range("2024-05-10", periods=3, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {"H": [17_249.0] * 3, "D": [-258.0] * 3, "Z": [45_900.0] * 3},
        index=index,
    )
    out = hdz_to_xyz(frame)
    assert list(out.columns)[:3] == ["X", "Y", "Z"]
    magnitude = np.sqrt(out["X"] ** 2 + out["Y"] ** 2)
    assert magnitude.iloc[0] == pytest.approx(17_249.0)
    assert out["Y"].iloc[0] < 0  # negative declination gives negative Y


def test_hdz_to_xyz_rejects_a_frame_without_h_and_d() -> None:
    with pytest.raises(ValueError, match="no H and D columns"):
        hdz_to_xyz(make_frame())


# --------------------------------------------------------------------
# End to end over the parser fixtures
# --------------------------------------------------------------------


def test_defective_fixture_raises_cadence_and_completeness_issues() -> None:
    parsed = parse_file(FIXTURES / "esk_defective_min.min")
    report = validate(parsed.data)
    assert not report.ok
    cadence = report.by_check("cadence")
    assert sum(i.count for i in cadence) == 4
    assert {i.element for i in report.by_check("completeness")} == {"X", "Y"}


def test_partial_fixture_reports_the_unobserved_element() -> None:
    parsed = parse_file(FIXTURES / "had_partial_min.min")
    report = validate(parsed.data)
    absent = report.by_check("completeness")
    assert [i.element for i in absent] == ["F"]
    assert absent[0].severity == "error"


def test_report_helpers_partition_by_severity() -> None:
    issues = (
        Issue(check="a", severity="error", message="x"),
        Issue(check="b", severity="warning", message="y"),
    )
    report = Report(issues=issues)
    assert len(report.errors) == 1
    assert len(report.warnings) == 1
    assert not report.ok
    assert "2 issue(s)" in report.summary()
