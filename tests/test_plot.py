"""Tests for :mod:`magpipe.plot`.

Plots are awkward to test because the interesting output is an image.
These tests do not compare pixels, which is brittle and slow. They
inspect the figure object instead: how many axes it has, what the lines
carry, what the labels say. That catches the failures which actually
occur -- an element silently dropped, a baseline subtracted twice, a gap
drawn as a straight line -- without depending on font rendering.

The Agg backend is selected so nothing requires a display.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from magpipe.plot import (  # noqa: E402
    plot_comparison,
    plot_residual,
    plot_timeseries,
    save,
)


def make_frame(n: int = 120, x: float = 17_200.0) -> pd.DataFrame:
    index = pd.date_range(
        "2024-05-10T00:00:00Z", periods=n, freq="1min", tz="UTC", name="time"
    )
    frame = pd.DataFrame(
        {
            "X": x + np.sin(np.arange(n) / 10.0) * 30.0,
            "Y": -1_300.0 + np.cos(np.arange(n) / 10.0) * 8.0,
            "Z": np.full(n, 45_900.0),
        },
        index=index,
    )
    frame["F"] = np.sqrt(
        frame["X"] ** 2 + frame["Y"] ** 2 + frame["Z"] ** 2
    ) + np.random.default_rng(0).normal(0.1, 0.4, n)
    return frame


# --------------------------------------------------------------------
# Time series
# --------------------------------------------------------------------


def test_one_panel_per_element() -> None:
    figure = plot_timeseries(make_frame())
    assert len(figure.axes) == 4


def test_elements_argument_selects_and_orders_panels() -> None:
    figure = plot_timeseries(make_frame(), elements=["Z", "X"])
    assert len(figure.axes) == 2
    assert figure.axes[0].get_ylabel().startswith("Z")
    assert figure.axes[1].get_ylabel().startswith("X")


def test_unknown_element_is_rejected() -> None:
    with pytest.raises(ValueError, match="has no column"):
        plot_timeseries(make_frame(), elements=["Q"])


def test_empty_element_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="no elements to plot"):
        plot_timeseries(make_frame(), elements=[])


def test_every_sample_reaches_the_line() -> None:
    frame = make_frame(n=200)
    figure = plot_timeseries(frame, elements=["X"], shade_gaps=False)
    line = figure.axes[0].lines[0]
    assert len(line.get_ydata()) == 200


def test_baseline_subtraction_centres_the_series() -> None:
    frame = make_frame()
    figure = plot_timeseries(frame, elements=["X"], shade_gaps=False)
    ydata = figure.axes[0].lines[0].get_ydata()
    assert abs(np.nanmedian(ydata)) < 1e-9
    assert abs(np.nanmax(ydata)) < 100.0


def test_baseline_offset_is_recorded_in_the_label() -> None:
    figure = plot_timeseries(make_frame(), elements=["X"], shade_gaps=False)
    label = figure.axes[0].get_ylabel()
    assert label.startswith("X ")
    assert "1720" in label.replace(",", "")
    assert "(nT)" in label


def test_baseline_subtraction_can_be_turned_off() -> None:
    figure = plot_timeseries(make_frame(), elements=["X"], subtract_baseline=False)
    ydata = figure.axes[0].lines[0].get_ydata()
    assert np.nanmedian(ydata) > 17_000.0
    assert figure.axes[0].get_ylabel() == "X (nT)"


def test_absent_values_do_not_become_zeros() -> None:
    frame = make_frame()
    frame.iloc[10:20, frame.columns.get_loc("X")] = np.nan
    figure = plot_timeseries(frame, elements=["X"], shade_gaps=False)
    ydata = figure.axes[0].lines[0].get_ydata()
    assert np.isnan(ydata[10:20]).all()


def test_gaps_are_shaded() -> None:
    frame = make_frame()
    frame = frame.drop(index=frame.index[40:50])
    figure = plot_timeseries(frame, elements=["X"], shade_gaps=True)
    assert len(figure.axes[0].patches) >= 1


def test_no_shading_when_the_series_is_continuous() -> None:
    figure = plot_timeseries(make_frame(), elements=["X"], shade_gaps=True)
    assert len(figure.axes[0].patches) == 0


def test_shading_can_be_turned_off() -> None:
    frame = make_frame()
    frame = frame.drop(index=frame.index[40:50])
    figure = plot_timeseries(frame, elements=["X"], shade_gaps=False)
    assert len(figure.axes[0].patches) == 0


def test_title_is_applied() -> None:
    figure = plot_timeseries(make_frame(), title="ESK 2024-05-10")
    assert figure._suptitle.get_text() == "ESK 2024-05-10"


def test_an_existing_figure_is_used() -> None:
    figure = Figure()
    assert plot_timeseries(make_frame(), figure=figure) is figure


# --------------------------------------------------------------------
# Residual
# --------------------------------------------------------------------


def test_residual_needs_all_four_elements() -> None:
    with pytest.raises(ValueError, match="missing"):
        plot_residual(make_frame().drop(columns=["F"]))


def test_residual_is_scalar_minus_vector() -> None:
    frame = make_frame()
    figure = plot_residual(frame)
    ydata = figure.axes[0].lines[0].get_ydata()
    expected = frame["F"] - np.sqrt(frame["X"] ** 2 + frame["Y"] ** 2 + frame["Z"] ** 2)
    assert np.allclose(ydata, expected.to_numpy())


def test_residual_title_reports_the_statistics() -> None:
    title = plot_residual(make_frame()).axes[0].get_title()
    assert "median" in title
    assert "sd" in title


def test_residual_marks_the_tolerance_band() -> None:
    figure = plot_residual(make_frame())
    assert len(figure.axes[0].patches) >= 1


# --------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------


def test_comparison_draws_one_line_per_observatory() -> None:
    frames = {"ESK": make_frame(), "HAD": make_frame(x=19_500.0)}
    figure = plot_comparison(frames, element="X")
    assert len(figure.axes[0].lines) == 2


def test_comparison_removes_the_site_baselines() -> None:
    """Two sites with very different absolute fields must overlay."""
    frames = {"ESK": make_frame(), "HAD": make_frame(x=19_500.0)}
    figure = plot_comparison(frames, element="X")
    medians = [np.nanmedian(line.get_ydata()) for line in figure.axes[0].lines]
    assert all(abs(m) < 1e-9 for m in medians)


def test_comparison_labels_carry_the_offsets() -> None:
    frames = {"ESK": make_frame(), "HAD": make_frame(x=19_500.0)}
    labels = [line.get_label() for line in plot_comparison(frames).axes[0].lines]
    assert any("ESK" in label for label in labels)
    assert any("1950" in label.replace(",", "") for label in labels)


def test_comparison_rejects_an_element_a_site_lacks() -> None:
    frames = {"ESK": make_frame(), "HAD": make_frame().drop(columns=["Z"])}
    with pytest.raises(ValueError, match="HAD does not report"):
        plot_comparison(frames, element="Z")


def test_comparison_rejects_an_empty_mapping() -> None:
    with pytest.raises(ValueError, match="no frames"):
        plot_comparison({})


# --------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------


def test_save_creates_the_parent_directory(tmp_path) -> None:
    target = tmp_path / "figures" / "esk.png"
    save(plot_timeseries(make_frame(), elements=["X"]), target)
    assert target.exists()
    assert target.stat().st_size > 0
