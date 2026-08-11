"""Tests for :mod:`magpipe.fetch`.

No test here touches the network. HTTP is exercised through a stub
session that records the URLs it is given and returns canned responses,
which is what lets the retry, caching and payload-validation logic be
tested deterministically.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from magpipe.fetch import (
    GIN_ENDPOINT,
    DataRequest,
    FetchError,
    PermanentFetchError,
    build_url,
    cache_path,
    fetch,
    fetch_many,
)

VALID_PAYLOAD = (
    " Format                 IAGA-2002                                    |\n"
    " Source of Data         British Geological Survey                    |\n"
    " IAGA CODE              ESK                                          |\n"
    "DATE       TIME         DOY     ESKX      ESKY      ESKZ      ESKF   |\n"
    "2024-05-10 00:00:00.000 131      17200.00  -1300.00  45900.00  49100.00\n"
)


class StubResponse:
    def __init__(self, status_code: int = 200, text: str = VALID_PAYLOAD) -> None:
        self.status_code = status_code
        self.text = text


class StubSession:
    """Returns queued responses and records the URLs requested."""

    def __init__(self, *responses: StubResponse | Exception) -> None:
        self._queue = list(responses) or [StubResponse()]
        self.urls: list[str] = []

    def get(self, url: str, timeout: float, headers: dict[str, str]):
        self.urls.append(url)
        item = self._queue.pop(0) if len(self._queue) > 1 else self._queue[0]
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def request_() -> DataRequest:
    return DataRequest(observatory="esk", start=date(2024, 5, 10))


def no_sleep(_seconds: float) -> None:
    return None


# --------------------------------------------------------------------
# Request construction
# --------------------------------------------------------------------


def test_observatory_code_is_normalised(request_: DataRequest) -> None:
    assert request_.code == "ESK"


def test_invalid_observatory_code_is_rejected() -> None:
    with pytest.raises(ValueError, match="three-letter IAGA code"):
        DataRequest(observatory="ESKD", start=date(2024, 5, 10))


def test_zero_days_is_rejected() -> None:
    with pytest.raises(ValueError, match="days must be at least 1"):
        DataRequest(observatory="ESK", start=date(2024, 5, 10), days=0)


def test_span_beyond_the_service_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="at most 31 days"):
        DataRequest(
            observatory="ESK", start=date(2024, 5, 10), days=40, cadence="Second"
        )
    # The same span is fine at minute cadence.
    DataRequest(observatory="ESK", start=date(2024, 5, 10), days=40)


def test_unknown_publication_state_is_rejected() -> None:
    with pytest.raises(ValueError, match="publication_state"):
        DataRequest(
            observatory="ESK", start=date(2024, 5, 10), publication_state="final"
        )


def test_unknown_orientation_is_rejected() -> None:
    with pytest.raises(ValueError, match="orientation"):
        DataRequest(observatory="ESK", start=date(2024, 5, 10), orientation="XYZQ")


def test_dates_are_sent_as_plain_days_not_timestamps(
    request_: DataRequest,
) -> None:
    params = request_.params()
    assert params["dataStartDate"] == "2024-05-10"
    assert "T" not in params["dataStartDate"]
    assert "dataEndDate" not in params


def test_duration_is_sent_rather_than_an_end_date() -> None:
    request = DataRequest(observatory="ESK", start=date(2024, 5, 10), days=3)
    assert request.params()["dataDuration"] == "3"
    assert request.end_date == date(2024, 5, 12)


def test_a_datetime_start_is_reduced_to_its_date() -> None:
    request = DataRequest(
        observatory="ESK", start=datetime(2024, 5, 10, 6, 30, tzinfo=UTC)
    )
    assert request.params()["dataStartDate"] == "2024-05-10"


def test_request_and_format_are_capitalised(request_: DataRequest) -> None:
    """The service returns HTTP 400 when these two are lower case."""
    params = request_.params()
    assert params["Request"] == "GetData"
    assert params["Format"] == "iaga2002"
    assert "request" not in params
    assert "format" not in params


def test_cadence_spellings_are_accepted() -> None:
    for cadence in ("minute", "Minute", 1440):
        request = DataRequest(
            observatory="ESK", start=date(2024, 5, 10), cadence=cadence
        )
        assert request.params()["samplesPerDay"] == "Minute"
    request = DataRequest(observatory="ESK", start=date(2024, 5, 10), cadence="second")
    assert request.params()["samplesPerDay"] == "Second"


def test_unknown_cadence_is_rejected() -> None:
    with pytest.raises(ValueError, match="cadence must be"):
        DataRequest(observatory="ESK", start=date(2024, 5, 10), cadence="hourly")


def test_url_carries_every_parameter(request_: DataRequest) -> None:
    url = build_url(request_)
    assert url.startswith(GIN_ENDPOINT + "?")
    for fragment in (
        "Request=GetData",
        "Format=iaga2002",
        "observatoryIagaCode=ESK",
        "samplesPerDay=Minute",
        "dataStartDate=2024-05-10",
        "dataDuration=1",
        "publicationState=best-avail",
        "orientation=Native",
    ):
        assert fragment in url


def test_slug_is_filesystem_safe(request_: DataRequest) -> None:
    slug = request_.slug()
    assert ":" not in slug
    assert "/" not in slug
    assert slug.startswith("esk_20240510_1d")


def test_cache_paths_differ_when_parameters_differ(
    request_: DataRequest, tmp_path: Path
) -> None:
    other = DataRequest(observatory="esk", start=request_.start, orientation="XYZS")
    assert cache_path(request_, tmp_path) != cache_path(other, tmp_path)


def test_cache_path_is_stable_for_the_same_request(
    request_: DataRequest, tmp_path: Path
) -> None:
    assert cache_path(request_, tmp_path) == cache_path(request_, tmp_path)


# --------------------------------------------------------------------
# Retrieval and caching
# --------------------------------------------------------------------


def test_successful_fetch_writes_the_cache(
    request_: DataRequest, tmp_path: Path
) -> None:
    session = StubSession()
    path = fetch(request_, session, tmp_path, sleep=no_sleep)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == VALID_PAYLOAD
    assert len(session.urls) == 1


def test_second_fetch_uses_the_cache(request_: DataRequest, tmp_path: Path) -> None:
    session = StubSession()
    fetch(request_, session, tmp_path, sleep=no_sleep)
    fetch(request_, session, tmp_path, sleep=no_sleep)
    assert len(session.urls) == 1


def test_refresh_forces_a_new_download(request_: DataRequest, tmp_path: Path) -> None:
    session = StubSession()
    fetch(request_, session, tmp_path, sleep=no_sleep)
    fetch(request_, session, tmp_path, refresh=True, sleep=no_sleep)
    assert len(session.urls) == 2


def test_user_agent_is_sent(request_: DataRequest, tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    class Recording(StubSession):
        def get(self, url: str, timeout: float, headers: dict[str, str]):
            seen.update(headers)
            return super().get(url, timeout, headers)

    fetch(request_, Recording(), tmp_path, sleep=no_sleep)
    assert "magpipe" in seen["User-Agent"]


# --------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------


def test_server_error_is_retried_then_raised(
    request_: DataRequest, tmp_path: Path
) -> None:
    session = StubSession(StubResponse(status_code=500))
    with pytest.raises(FetchError, match="failed after 3 attempts"):
        fetch(request_, session, tmp_path, sleep=no_sleep)
    assert len(session.urls) == 3


def test_bad_request_is_not_retried(request_: DataRequest, tmp_path: Path) -> None:
    """HTTP 400 means the URL is wrong; retrying cannot fix it."""
    session = StubSession(StubResponse(status_code=400))
    with pytest.raises(PermanentFetchError, match="bad request parameters"):
        fetch(request_, session, tmp_path, sleep=no_sleep)
    assert len(session.urls) == 1


def test_missing_data_is_not_retried(request_: DataRequest, tmp_path: Path) -> None:
    session = StubSession(StubResponse(status_code=404))
    with pytest.raises(PermanentFetchError, match="no data available"):
        fetch(request_, session, tmp_path, sleep=no_sleep)
    assert len(session.urls) == 1


def test_permanent_error_message_includes_the_url(
    request_: DataRequest, tmp_path: Path
) -> None:
    session = StubSession(StubResponse(status_code=400))
    with pytest.raises(PermanentFetchError, match="GINServices"):
        fetch(request_, session, tmp_path, sleep=no_sleep)


def test_transient_failure_then_success(request_: DataRequest, tmp_path: Path) -> None:
    session = StubSession(StubResponse(status_code=503), StubResponse())
    path = fetch(request_, session, tmp_path, sleep=no_sleep)
    assert path.exists()
    assert len(session.urls) == 2


def test_html_response_is_rejected(request_: DataRequest, tmp_path: Path) -> None:
    session = StubSession(StubResponse(text="<html><body>no data</body></html>"))
    with pytest.raises(FetchError):
        fetch(request_, session, tmp_path, retries=1, sleep=no_sleep)


def test_empty_response_is_rejected(request_: DataRequest, tmp_path: Path) -> None:
    session = StubSession(StubResponse(text="   \n"))
    with pytest.raises(FetchError):
        fetch(request_, session, tmp_path, retries=1, sleep=no_sleep)


def test_non_iaga_response_is_rejected(request_: DataRequest, tmp_path: Path) -> None:
    session = StubSession(StubResponse(text="observatory not found"))
    with pytest.raises(FetchError):
        fetch(request_, session, tmp_path, retries=1, sleep=no_sleep)


def test_a_rejected_payload_is_not_cached(
    request_: DataRequest, tmp_path: Path
) -> None:
    session = StubSession(StubResponse(text="observatory not found"))
    with pytest.raises(FetchError):
        fetch(request_, session, tmp_path, retries=1, sleep=no_sleep)
    assert not cache_path(request_, tmp_path).exists()


def test_connection_error_is_retried(request_: DataRequest, tmp_path: Path) -> None:
    session = StubSession(ConnectionError("refused"), StubResponse())
    path = fetch(request_, session, tmp_path, sleep=no_sleep)
    assert path.exists()


def test_fetch_many_continues_past_a_failure(tmp_path: Path) -> None:
    good = DataRequest(observatory="ESK", start=date(2024, 5, 10))
    bad = DataRequest(observatory="XYZ", start=date(2024, 5, 10))

    class Selective:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def get(self, url: str, timeout: float, headers: dict[str, str]):
            self.urls.append(url)
            if "XYZ" in url:
                return StubResponse(status_code=404)
            return StubResponse()

    results = fetch_many(
        [good, bad], Selective(), cache_dir=tmp_path, retries=1, sleep=no_sleep
    )
    assert list(results) == [good]
