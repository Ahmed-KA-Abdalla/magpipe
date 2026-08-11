"""Integration tests against real INTERMAGNET observatory data.

These tests are skipped unless ``tests/data/`` contains extracts produced
by ``scripts/make_sample.py``. They exist because synthetic fixtures test
what the format specification says, whereas real files test what the
service actually sends -- header padding, element sets, publication
states and encoding all vary in practice.

To populate them::

    python scripts/fetch_data.py --obs ESK --start 2024-05-10 --end 2024-05-10
    python scripts/make_sample.py data/raw/esk_*.min --hours 6 --label esk_storm

The extracts carry the INTERMAGNET acknowledgement as comment records.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from magpipe.parse import parse_file
from magpipe.validate import Thresholds, validate

DATA = Path(__file__).parent / "data"
SAMPLES = sorted(DATA.glob("*.min")) if DATA.exists() else []

pytestmark = pytest.mark.skipif(
    not SAMPLES,
    reason="no real extracts in tests/data/; see scripts/make_sample.py",
)


@pytest.fixture(params=SAMPLES, ids=lambda p: p.stem)
def sample(request) -> Path:
    return request.param


def test_real_file_parses(sample: Path) -> None:
    parsed = parse_file(sample)
    assert len(parsed) > 0
    assert parsed.iaga_code.isalpha()
    assert len(parsed.iaga_code) == 3


def test_real_file_declares_the_expected_headers(sample: Path) -> None:
    parsed = parse_file(sample)
    assert parsed.header["Format"] == "IAGA-2002"
    assert parsed.header["Data Type"]
    assert -90.0 <= parsed.latitude <= 90.0
    assert 0.0 <= parsed.longitude <= 360.0


def test_real_file_carries_the_acknowledgement(sample: Path) -> None:
    comments = " ".join(parse_file(sample).comments)
    assert "INTERMAGNET" in comments


def test_real_file_index_is_regular_and_utc(sample: Path) -> None:
    data = parse_file(sample).data
    assert str(data.index.tz) == "UTC"
    assert data.index.is_monotonic_increasing
    assert not data.index.has_duplicates


def test_real_file_values_are_physically_plausible(sample: Path) -> None:
    """The range check should pass on definitive observatory data."""
    report = validate(parse_file(sample).data)
    assert report.by_check("range") == ()


def test_real_file_is_broadly_complete(sample: Path) -> None:
    data = parse_file(sample).data
    reported = [c for c in data.columns if not data[c].isna().all()]
    assert reported, "every element is absent"
    for element in reported:
        assert data[element].isna().mean() < 0.10


def test_real_file_scalar_and_vector_field_agree(sample: Path) -> None:
    """Definitive data should pass the consistency check at its defaults.

    The defaults were measured from ESK and HAD on 2024-05-10; see the
    note on Thresholds. A failure here means either a genuine instrument
    problem in the file or that the defaults are too tight for this
    observatory, and both are worth knowing.
    """
    data = parse_file(sample).data
    if not {"X", "Y", "Z", "F"}.issubset(data.columns):
        pytest.skip("file reports G rather than F; use --orientation XYZS")
    assert validate(data).by_check("field_consistency") == ()


def test_real_file_cadence_matches_its_declared_interval(sample: Path) -> None:
    parsed = parse_file(sample)
    declared = parsed.header["Data Interval Type"].lower()
    if "minute" in declared:
        expected = pd.Timedelta("1min")
    elif "second" in declared:
        expected = pd.Timedelta("1s")
    else:
        pytest.skip(f"unhandled interval type: {declared!r}")
    report = validate(parsed.data, Thresholds(expected_interval=expected))
    gaps = sum(i.count for i in report.by_check("cadence"))
    assert gaps < 0.02 * len(parsed)
