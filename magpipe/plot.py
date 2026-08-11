"""Plots of magnetometer time series.

Three figures are provided:

:func:`plot_timeseries`
    One panel per element, sharing a time axis, with data gaps shaded.
:func:`plot_residual`
    The difference between the scalar and vector total field, which is
    the quantity :func:`magpipe.validate.check_field_consistency`
    thresholds.
:func:`plot_comparison`
    One element from several observatories overlaid, for seeing whether
    a disturbance is global or local.

Two conventions are worth stating, since both affect what a reader
takes from the figure.

Baselines are subtracted by default. The components are of order tens
of thousands of nanotesla while the variation of interest is tens to
hundreds, so plotting absolute values renders the signal invisible. The
median over the plotted interval is removed and the offset is recorded
in the panel label, so nothing is hidden.

Gaps are shaded rather than interpolated across. Matplotlib joins the
points either side of an absent sample with a straight line, which
misrepresents an outage as a smooth trend. Absent intervals are found
by :func:`magpipe.validate.check_cadence` and marked.

Every function takes an explicit :class:`~matplotlib.figure.Figure` or
creates one and returns it; none calls ``show`` or ``savefig``, so the
caller decides what happens to the result and the tests can inspect the
figure directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from magpipe.validate import Thresholds, check_cadence

__all__ = [
    "ELEMENT_LABELS",
    "plot_timeseries",
    "plot_residual",
    "plot_comparison",
    "save",
]

#: Axis labels by element code. D and I are angles, the rest are fields.
ELEMENT_LABELS = {
    "X": "X (nT)",
    "Y": "Y (nT)",
    "Z": "Z (nT)",
    "H": "H (nT)",
    "F": "F (nT)",
    "S": "S (nT)",
    "G": "G (nT)",
    "D": "D (arcmin)",
    "I": "I (arcmin)",
}


def plot_timeseries(
    frame: pd.DataFrame,
    title: str = "",
    elements: Sequence[str] | None = None,
    *,
    subtract_baseline: bool = True,
    shade_gaps: bool = True,
    thresholds: Thresholds | None = None,
    figure: Figure | None = None,
) -> Figure:
    """One panel per element, sharing a time axis.

    Parameters
    ----------
    frame:
        Output of :func:`magpipe.parse.parse_text`.
    elements:
        Which columns to draw, in order. Defaults to every column.
    subtract_baseline:
        Remove the median of each element over the plotted interval and
        record it in the panel label.
    shade_gaps:
        Mark intervals where samples are absent.
    """
    columns = list(elements) if elements is not None else list(frame.columns)
    _check_columns(frame, columns)
    if not columns:
        raise ValueError("no elements to plot")

    figure = figure or Figure(figsize=(10, 2.0 * len(columns) + 1.2))
    axes = figure.subplots(len(columns), 1, sharex=True, squeeze=False)[:, 0]

    gaps = check_cadence(frame, thresholds or Thresholds()) if shade_gaps else []

    for axis, element in zip(axes, columns, strict=True):
        series = frame[element]
        label = ELEMENT_LABELS.get(element, f"{element} (nT)")
        if subtract_baseline and series.notna().any():
            offset = float(series.median())
            series = series - offset
            label = f"{element} − {offset:.0f} (nT)"
        axis.plot(series.index, series.to_numpy(), linewidth=0.8)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        for issue in gaps:
            if issue.start is not None and issue.end is not None:
                axis.axvspan(issue.start, issue.end, alpha=0.25, zorder=0)

    _format_time_axis(axes[-1], frame.index)
    if title:
        figure.suptitle(title)
    figure.align_ylabels(axes)
    figure.tight_layout()
    return figure


def plot_residual(
    frame: pd.DataFrame,
    title: str = "",
    *,
    thresholds: Thresholds | None = None,
    figure: Figure | None = None,
) -> Figure:
    """Scalar minus vector total field, with the tolerance band marked.

    Requires X, Y, Z and F. Under an ``XYZS`` orientation F comes from
    an independent instrument and the residual is informative; under
    ``XYZF`` it is identically zero by construction.
    """
    required = {"X", "Y", "Z", "F"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"residual needs X, Y, Z and F; missing {sorted(missing)}")

    thresholds = thresholds or Thresholds()
    vector = np.sqrt(frame["X"] ** 2 + frame["Y"] ** 2 + frame["Z"] ** 2)
    residual = frame["F"] - vector

    figure = figure or Figure(figsize=(10, 4))
    axis = figure.subplots()
    axis.plot(residual.index, residual.to_numpy(), linewidth=0.8)
    axis.axhline(0.0, linewidth=0.8, alpha=0.5)
    limit = thresholds.max_field_residual_nt
    axis.axhspan(-limit, limit, alpha=0.15, zorder=0)
    axis.set_ylabel("F − |(X, Y, Z)| (nT)")
    axis.grid(True, alpha=0.3)

    finite = residual.dropna()
    if not finite.empty:
        axis.set_title(
            f"median {finite.median():+.2f} nT, "
            f"sd {finite.std():.2f} nT, "
            f"largest |residual| {finite.abs().max():.2f} nT",
            fontsize="small",
        )
    _format_time_axis(axis, frame.index)
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_comparison(
    frames: Mapping[str, pd.DataFrame],
    element: str = "X",
    title: str = "",
    *,
    subtract_baseline: bool = True,
    figure: Figure | None = None,
) -> Figure:
    """One element from several observatories, overlaid on one axis."""
    if not frames:
        raise ValueError("no frames to compare")

    figure = figure or Figure(figsize=(10, 4.5))
    axis = figure.subplots()

    for name, frame in frames.items():
        if element not in frame.columns:
            raise ValueError(f"{name} does not report element {element!r}")
        series = frame[element]
        label = name
        if subtract_baseline and series.notna().any():
            offset = float(series.median())
            series = series - offset
            label = f"{name} (−{offset:.0f} nT)"
        axis.plot(series.index, series.to_numpy(), linewidth=0.8, label=label)

    axis.set_ylabel(
        f"{element} anomaly (nT)"
        if subtract_baseline
        else ELEMENT_LABELS.get(element, f"{element} (nT)")
    )
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small")
    first = next(iter(frames.values()))
    _format_time_axis(axis, first.index)
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def save(figure: Figure, path, dpi: int = 150) -> None:
    """Write a figure to disk, creating the parent directory."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")


def _check_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    unknown = [c for c in columns if c not in frame.columns]
    if unknown:
        raise ValueError(
            f"frame has no column(s) {unknown}; it reports {list(frame.columns)}"
        )


def _format_time_axis(axis, index: pd.DatetimeIndex) -> None:
    """Label the shared time axis, choosing a locator to suit the span."""
    axis.set_xlabel("UTC")
    if len(index) == 0:
        return
    span = index.max() - index.min()
    if span <= pd.Timedelta("36h"):
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
