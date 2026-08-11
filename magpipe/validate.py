"""Quality checks over parsed magnetometer data.

The checks here answer the question a data pipeline has to answer before
anything downstream trusts its output: is this file fit to use, and if
not, what is wrong with it and where.

Each check is a pure function taking a frame and a :class:`Thresholds`
instance and returning a list of :class:`Issue`. :func:`validate` runs
all of them and returns a :class:`Report`. No check raises on bad data;
defects are reported, not thrown, because a pipeline that aborts on the
first bad sample cannot process an archive.

Checks implemented:

``cadence``
    Samples absent from an otherwise regular series, reported as
    intervals rather than as individual timestamps.
``ordering``
    Duplicated or out-of-order timestamps.
``completeness``
    Fraction of absent values per element against a tolerance.
``range``
    Values outside the plausible range for a surface observatory.
``spikes``
    Sample-to-sample changes too large to be geophysical.
``field_consistency``
    Disagreement between the scalar field F and the magnitude computed
    from the vector components. This is the check that catches real
    instrument faults, since F is measured independently.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "Issue",
    "Report",
    "Thresholds",
    "Severity",
    "validate",
    "check_cadence",
    "check_ordering",
    "check_completeness",
    "check_range",
    "check_spikes",
    "check_field_consistency",
    "hdz_to_xyz",
]

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Thresholds:
    """Tunable limits for the checks.

    Defaults are chosen for one-minute data from a mid-latitude surface
    observatory. They are deliberately loose: the purpose is to catch
    instrument faults and file corruption, not to reject genuine
    geomagnetic activity. A severe storm can exceed 100 nT/min, so
    ``max_step_nt`` is set well above that.
    """

    expected_interval: pd.Timedelta = pd.Timedelta("1min")
    #: Plausible magnitude of any single component at the surface.
    max_component_nt: float = 70_000.0
    #: Plausible range for the total field magnitude at the surface.
    min_total_nt: float = 20_000.0
    max_total_nt: float = 70_000.0
    #: Largest credible change between consecutive one-minute samples.
    max_step_nt: float = 500.0
    #: Tolerances for the scalar/vector field comparison. Measured from
    #: definitive XYZS data for ESK and HAD on 2024-05-10: the residual
    #: has a median near zero (+0.13 nT at ESK, +0.19 at HAD), a
    #: standard deviation of about 0.5 nT, and individual excursions to
    #: roughly 5 nT. A systematic offset therefore indicates a baseline
    #: problem, whereas isolated excursions are ordinary instrument
    #: disagreement and are only worth reporting in quantity.
    #:
    #: Largest tolerated systematic offset, taken as the median residual.
    max_field_offset_nt: float = 1.0
    #: Largest tolerated residual on an individual sample.
    max_field_residual_nt: float = 5.0
    #: Fraction of samples allowed to exceed the residual limit.
    max_field_excursion_fraction: float = 0.01
    #: Fraction of absent values above which an element is unusable.
    max_absent_fraction: float = 0.05


@dataclass(frozen=True)
class Issue:
    """A single quality finding."""

    check: str
    severity: Severity
    message: str
    element: str | None = None
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    count: int = 1

    def as_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "element": self.element,
            "start": self.start,
            "end": self.end,
            "count": self.count,
            "message": self.message,
        }


@dataclass(frozen=True)
class Report:
    """The outcome of running every check over one frame."""

    issues: tuple[Issue, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """True when no error-severity issue was raised."""
        return not self.errors

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    def by_check(self, name: str) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.check == name)

    def to_frame(self) -> pd.DataFrame:
        """Tabular form, for writing to a database or a test report."""
        if not self.issues:
            return pd.DataFrame(
                columns=[
                    "check",
                    "severity",
                    "element",
                    "start",
                    "end",
                    "count",
                    "message",
                ]
            )
        return pd.DataFrame([i.as_dict() for i in self.issues])

    def summary(self) -> str:
        if not self.issues:
            return "No issues."
        counts = self.to_frame().groupby(["check", "severity"]).size().sort_index()
        lines = [f"{len(self.issues)} issue(s):"]
        for (check, severity), n in counts.items():
            lines.append(f"  {check} ({severity}): {n}")
        return "\n".join(lines)


def hdz_to_xyz(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert H, D, Z columns to X, Y, Z.

    D is the declination, recorded in minutes of arc in IAGA-2002 files.
    Columns other than H and D are carried through unchanged.
    """
    if not {"H", "D"}.issubset(frame.columns):
        raise ValueError("frame has no H and D columns to convert")
    declination = np.deg2rad(frame["D"] / 60.0)
    out = frame.drop(columns=["H", "D"]).copy()
    out["X"] = frame["H"] * np.cos(declination)
    out["Y"] = frame["H"] * np.sin(declination)
    ordered = [c for c in ("X", "Y", "Z", "F") if c in out.columns]
    return out[ordered + [c for c in out.columns if c not in ordered]]


def check_ordering(frame: pd.DataFrame, thresholds: Thresholds) -> list[Issue]:
    issues: list[Issue] = []
    index = frame.index
    if len(index) < 2:
        return issues

    duplicated = index[index.duplicated()]
    if len(duplicated):
        issues.append(
            Issue(
                check="ordering",
                severity="error",
                message=f"{len(duplicated)} duplicated timestamp(s)",
                start=duplicated.min(),
                end=duplicated.max(),
                count=len(duplicated),
            )
        )
    if not index.is_monotonic_increasing:
        issues.append(
            Issue(
                check="ordering",
                severity="error",
                message="timestamps are not in increasing order",
                start=index.min(),
                end=index.max(),
            )
        )
    return issues


def check_cadence(frame: pd.DataFrame, thresholds: Thresholds) -> list[Issue]:
    """Report intervals where samples are absent from the series."""
    issues: list[Issue] = []
    index = frame.index
    if len(index) < 2 or not index.is_monotonic_increasing:
        return issues

    steps = index.to_series().diff().dropna()
    interval = thresholds.expected_interval
    gap_ends = steps[steps > interval]

    for end, step in gap_ends.items():
        n_absent = int(step / interval) - 1
        issues.append(
            Issue(
                check="cadence",
                severity="error",
                message=(
                    f"{n_absent} sample(s) absent over {step} "
                    f"at the expected {interval} cadence"
                ),
                start=end - step + interval,
                end=end - interval,
                count=n_absent,
            )
        )

    irregular = steps[steps % interval != pd.Timedelta(0)]
    if len(irregular):
        issues.append(
            Issue(
                check="cadence",
                severity="error",
                message=(
                    f"{len(irregular)} interval(s) are not a whole multiple "
                    f"of the expected {interval} cadence"
                ),
                start=irregular.index.min(),
                end=irregular.index.max(),
                count=len(irregular),
            )
        )
    return issues


def check_completeness(frame: pd.DataFrame, thresholds: Thresholds) -> list[Issue]:
    issues: list[Issue] = []
    if frame.empty:
        return [
            Issue(
                check="completeness",
                severity="error",
                message="frame contains no samples",
                count=0,
            )
        ]

    for element in frame.columns:
        absent = frame[element].isna()
        fraction = float(absent.mean())
        if fraction == 0.0:
            continue
        severity: Severity = (
            "error" if fraction > thresholds.max_absent_fraction else "warning"
        )
        stamps = frame.index[absent]
        issues.append(
            Issue(
                check="completeness",
                severity=severity,
                element=element,
                message=(
                    f"{int(absent.sum())} of {len(frame)} values absent "
                    f"({fraction:.1%}), tolerance "
                    f"{thresholds.max_absent_fraction:.1%}"
                ),
                start=stamps.min(),
                end=stamps.max(),
                count=int(absent.sum()),
            )
        )
    return issues


def check_range(frame: pd.DataFrame, thresholds: Thresholds) -> list[Issue]:
    issues: list[Issue] = []
    for element in frame.columns:
        column = frame[element].dropna()
        if column.empty:
            continue
        if element == "F":
            out = column[
                (column < thresholds.min_total_nt) | (column > thresholds.max_total_nt)
            ]
            bound = f"[{thresholds.min_total_nt:.0f}, {thresholds.max_total_nt:.0f}]"
        else:
            out = column[column.abs() > thresholds.max_component_nt]
            bound = f"|value| <= {thresholds.max_component_nt:.0f}"
        if not out.empty:
            issues.append(
                Issue(
                    check="range",
                    severity="error",
                    element=element,
                    message=(
                        f"{len(out)} value(s) outside {bound} nT, "
                        f"extremes {out.min():.1f} to {out.max():.1f}"
                    ),
                    start=out.index.min(),
                    end=out.index.max(),
                    count=len(out),
                )
            )
    return issues


def check_spikes(frame: pd.DataFrame, thresholds: Thresholds) -> list[Issue]:
    """Flag sample-to-sample changes too large to be geophysical."""
    issues: list[Issue] = []
    for element in frame.columns:
        column = frame[element]
        steps = column.diff().abs()
        spikes = steps[steps > thresholds.max_step_nt].dropna()
        if not spikes.empty:
            issues.append(
                Issue(
                    check="spikes",
                    severity="warning",
                    element=element,
                    message=(
                        f"{len(spikes)} step(s) exceed "
                        f"{thresholds.max_step_nt:.0f} nT between consecutive "
                        f"samples, largest {spikes.max():.1f} nT"
                    ),
                    start=spikes.index.min(),
                    end=spikes.index.max(),
                    count=len(spikes),
                )
            )
    return issues


def check_field_consistency(frame: pd.DataFrame, thresholds: Thresholds) -> list[Issue]:
    """Compare the scalar field F against the vector magnitude.

    Under an ``XYZS`` or ``HDZS`` orientation, F comes from a separate
    scalar magnetometer, so agreement between the two is an independent
    check on both instruments. Under ``XYZF`` or ``HDZF`` the service
    computes F from the vector components and this check is a tautology;
    it will pass trivially, which is worth knowing when reading a report.

    Two distinct faults are separated here. A systematic offset -- a
    non-zero median residual -- points at a baseline error and is
    reported as an error. Isolated excursions are ordinary instrument
    disagreement; they are reported only when a meaningful fraction of
    samples is affected, and then as a warning.
    """
    required = {"X", "Y", "Z", "F"}
    if not required.issubset(frame.columns):
        return []

    vector = np.sqrt(frame["X"] ** 2 + frame["Y"] ** 2 + frame["Z"] ** 2)
    residual = (frame["F"] - vector).dropna()
    if residual.empty:
        return []

    issues: list[Issue] = []
    median = float(residual.median())
    if abs(median) > thresholds.max_field_offset_nt:
        issues.append(
            Issue(
                check="field_consistency",
                severity="error",
                element="F",
                message=(
                    f"systematic offset between scalar and vector field: "
                    f"median residual {median:+.2f} nT exceeds "
                    f"{thresholds.max_field_offset_nt:.2f} nT"
                ),
                start=residual.index.min(),
                end=residual.index.max(),
                count=len(residual),
            )
        )

    excursions = residual[residual.abs() > thresholds.max_field_residual_nt]
    fraction = len(excursions) / len(residual)
    if fraction > thresholds.max_field_excursion_fraction:
        issues.append(
            Issue(
                check="field_consistency",
                severity="warning",
                element="F",
                message=(
                    f"{len(excursions)} of {len(residual)} samples "
                    f"({fraction:.1%}) exceed "
                    f"{thresholds.max_field_residual_nt:.1f} nT, tolerance "
                    f"{thresholds.max_field_excursion_fraction:.1%}; "
                    f"largest {excursions.abs().max():.1f} nT"
                ),
                start=excursions.index.min(),
                end=excursions.index.max(),
                count=len(excursions),
            )
        )
    return issues


#: Every check, in the order :func:`validate` runs them.
CHECKS: tuple[Callable[[pd.DataFrame, Thresholds], list[Issue]], ...] = (
    check_ordering,
    check_cadence,
    check_completeness,
    check_range,
    check_spikes,
    check_field_consistency,
)


def validate(frame: pd.DataFrame, thresholds: Thresholds | None = None) -> Report:
    """Run every check and collect the findings."""
    thresholds = thresholds or Thresholds()
    issues: list[Issue] = []
    for check in CHECKS:
        issues.extend(check(frame, thresholds))
    return Report(issues=tuple(issues))
