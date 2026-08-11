"""Retrieval of observatory data from the INTERMAGNET GIN web service.

Data are served by the Edinburgh Geomagnetic Information Node, operated
by the British Geological Survey, from::

    https://imag-data.bgs.ac.uk/GIN_V1/GINServices

Requests are controlled entirely by URL query parameters, documented at
https://imag-data.bgs.ac.uk/GIN_V1/. Three details of that interface are
easy to get wrong and produce HTTP 400:

  * ``Request`` and ``Format`` are capitalised, while the remaining
    parameter names begin with a lower-case letter;
  * dates are plain ``yyyy-mm-dd``, not ISO 8601 timestamps, so the
    service works in whole days;
  * the interval is given either as ``dataEndDate`` or as
    ``dataDuration`` in days, not both.

This module builds those URLs, retrieves the response, and caches it on
disk so that repeated runs do not repeat the download. Nothing here
parses the payload; that is :mod:`magpipe.parse`.

Licensing
---------
INTERMAGNET data are distributed under CC-BY-NC and are subject to
INTERMAGNET's conditions of use at
https://intermagnet.org/data_conditions.html. Downloads land in a cache
directory excluded from version control; see ``scripts/make_sample.py``
for producing the small attributed extract the repository commits.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "GIN_ENDPOINT",
    "CADENCES",
    "ORIENTATIONS",
    "PUBLICATION_STATES",
    "FetchError",
    "PermanentFetchError",
    "DataRequest",
    "build_url",
    "fetch",
    "fetch_many",
    "cache_path",
]

logger = logging.getLogger(__name__)

GIN_ENDPOINT = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"

USER_AGENT = "magpipe/0.1 (portfolio project; contact via repository)"

#: Cadences the service accepts for ``samplesPerDay``. It also accepts
#: the equivalent integers, 1440 and 86400.
CADENCES = ("Minute", "Second")

#: Publication states. ``best-avail`` returns the most finished data the
#: provider has supplied, preferring definitive over quasi-definitive
#: over adjusted over reported.
PUBLICATION_STATES = (
    "best-avail",
    "definitive",
    "quasi-def",
    "adjusted",
    "reported",
)

#: Component orientations.
#:
#: The distinction that matters for quality checking is the final
#: letter. ``XYZF`` and ``HDZF`` compute the total field *from* the
#: vector components, which makes comparing F against |(X, Y, Z)| a
#: tautology. ``XYZS`` and ``HDZS`` take F from an independent scalar
#: instrument, which makes that comparison a real check on both
#: instruments. Use an S orientation if you want the consistency check
#: to mean anything. ``Native`` returns whatever the provider sent.
ORIENTATIONS = (
    "Native",
    "XYZF",
    "HDZF",
    "DIFF",
    "XYZS",
    "HDZS",
    "DIFS",
)

#: Service limits on the span of a single request.
MAX_DAYS = {"Minute": 366, "Second": 31}


class FetchError(RuntimeError):
    """Raised when the service returns something unusable."""


class PermanentFetchError(FetchError):
    """A failure that retrying cannot fix, such as a malformed request."""


class Response(Protocol):
    """The subset of a HTTP response this module relies on."""

    status_code: int
    text: str


class Session(Protocol):
    """The subset of :class:`requests.Session` this module relies on."""

    def get(self, url: str, timeout: float, headers: dict[str, str]) -> Response: ...


@dataclass(frozen=True)
class DataRequest:
    """One observatory over one interval at one cadence.

    Parameters
    ----------
    observatory:
        Three-letter IAGA code, for example ``ESK`` or ``HAD``.
    start:
        First day of data. Times are ignored; the service works in
        whole days.
    days:
        Number of days, sent as ``dataDuration``. Preferred over an end
        date because the service's treatment of ``dataEndDate`` at the
        boundary is ambiguous, whereas a duration is not.
    cadence:
        ``Minute`` or ``Second``.
    publication_state:
        One of :data:`PUBLICATION_STATES`.
    orientation:
        One of :data:`ORIENTATIONS`. See the note there on why an S
        orientation is the useful one for quality checking.
    """

    observatory: str
    start: date | datetime
    days: int = 1
    cadence: str = "Minute"
    publication_state: str = "best-avail"
    orientation: str = "Native"
    data_format: str = "iaga2002"
    record_termination: str = "UNIX"

    def __post_init__(self) -> None:
        if len(self.observatory) != 3 or not self.observatory.isalpha():
            raise ValueError(
                f"observatory must be a three-letter IAGA code, "
                f"got {self.observatory!r}"
            )
        if self.days < 1:
            raise ValueError(f"days must be at least 1, got {self.days}")
        cadence = _normalise_cadence(self.cadence)
        limit = MAX_DAYS[cadence]
        if self.days > limit:
            raise ValueError(
                f"the service allows at most {limit} days of "
                f"{cadence.lower()} data per request, asked for {self.days}"
            )
        if self.publication_state not in PUBLICATION_STATES:
            raise ValueError(
                f"publication_state must be one of {PUBLICATION_STATES}, "
                f"got {self.publication_state!r}"
            )
        if self.orientation not in ORIENTATIONS:
            raise ValueError(
                f"orientation must be one of {ORIENTATIONS}, got {self.orientation!r}"
            )

    @property
    def code(self) -> str:
        return self.observatory.upper()

    @property
    def start_date(self) -> date:
        return _as_date(self.start)

    @property
    def end_date(self) -> date:
        """Last day covered, inclusive."""
        return self.start_date + timedelta(days=self.days - 1)

    def params(self) -> dict[str, str]:
        return {
            "Request": "GetData",
            "Format": self.data_format,
            "observatoryIagaCode": self.code,
            "samplesPerDay": _normalise_cadence(self.cadence),
            "dataStartDate": self.start_date.isoformat(),
            "dataDuration": str(self.days),
            "publicationState": self.publication_state,
            "orientation": self.orientation,
            "recordTermination": self.record_termination,
        }

    def slug(self) -> str:
        """A stable, filesystem-safe name for this request."""
        return (
            f"{self.code.lower()}"
            f"_{self.start_date:%Y%m%d}"
            f"_{self.days}d"
            f"_{_normalise_cadence(self.cadence).lower()}"
            f"_{self.publication_state}"
        )


def build_url(request: DataRequest, endpoint: str = GIN_ENDPOINT) -> str:
    """Assemble the full request URL."""
    from urllib.parse import urlencode

    return f"{endpoint}?{urlencode(request.params())}"


def cache_path(request: DataRequest, cache_dir: Path) -> Path:
    """Where the response for this request is cached.

    The slug carries the main request parameters so that cached files
    are self-describing; a short hash of the complete parameter set is
    appended so that two requests differing only in a field absent from
    the slug cannot collide.
    """
    digest = hashlib.sha256(
        "&".join(f"{k}={v}" for k, v in sorted(request.params().items())).encode()
    ).hexdigest()[:8]
    return Path(cache_dir) / f"{request.slug()}_{digest}.min"


def fetch(
    request: DataRequest,
    session: Session,
    cache_dir: Path = Path("data/raw"),
    *,
    refresh: bool = False,
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 2.0,
    endpoint: str = GIN_ENDPOINT,
    sleep: Any = time.sleep,
) -> Path:
    """Retrieve one request, returning the path to the cached file.

    A cached file is returned untouched unless ``refresh`` is set.
    Transient failures are retried with exponential backoff, but a
    request the service rejects outright is not: HTTP 400 means the URL
    is malformed and will stay malformed, so retrying wastes the
    service's time and obscures the error. A response that does not look
    like IAGA-2002 is treated as an error rather than written to the
    cache, so a failed download cannot poison later runs.
    """
    path = cache_path(request, cache_dir)
    if path.exists() and not refresh:
        logger.info("cache hit: %s", path.name)
        return path

    url = build_url(request, endpoint=endpoint)
    last: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                url, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
            if response.status_code in (400, 404):
                reason = (
                    "bad request parameters"
                    if response.status_code == 400
                    else "no data available for this request"
                )
                raise PermanentFetchError(
                    f"{request.code}: HTTP {response.status_code} ({reason})\n  {url}"
                )
            if response.status_code != 200:
                raise FetchError(
                    f"{request.code}: service returned HTTP {response.status_code}"
                )
            text = response.text
            _check_payload(text, request)
        except PermanentFetchError:
            raise
        except Exception as exc:  # noqa: BLE001 - retried below
            last = exc
            if attempt == retries:
                break
            delay = backoff ** (attempt - 1)
            logger.warning(
                "%s: attempt %d/%d failed (%s); retrying in %.0fs",
                request.code,
                attempt,
                retries,
                exc,
                delay,
            )
            sleep(delay)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            logger.info(
                "retrieved %s (%d bytes) -> %s",
                request.code,
                len(text),
                path.name,
            )
            return path

    raise FetchError(
        f"{request.code}: download failed after {retries} attempts"
    ) from last


def fetch_many(
    requests: list[DataRequest],
    session: Session,
    cache_dir: Path = Path("data/raw"),
    **kwargs: Any,
) -> dict[DataRequest, Path]:
    """Retrieve several requests, continuing past individual failures.

    Returns the successful downloads. Failures are logged; the caller
    decides whether a partial result is acceptable.
    """
    out: dict[DataRequest, Path] = {}
    for request in requests:
        try:
            out[request] = fetch(request, session, cache_dir, **kwargs)
        except FetchError as exc:
            logger.error("%s", exc)
    return out


def _check_payload(text: str, request: DataRequest) -> None:
    """Reject anything that is not an IAGA-2002 file.

    The service answers some failures with an HTML page or an empty
    body under HTTP 200, so the status code alone is not sufficient.
    """
    head = text.lstrip()[:400]
    if not head:
        raise FetchError(f"{request.code}: service returned an empty body")
    if head.startswith("<"):
        raise FetchError(
            f"{request.code}: service returned markup, not data "
            f"(no data for the interval requested?)"
        )
    if "IAGA-2002" not in head:
        raise FetchError(
            f"{request.code}: response does not look like IAGA-2002; "
            f"first bytes: {head[:80]!r}"
        )


def _normalise_cadence(cadence: str | int) -> str:
    """Accept 'minute', 'Minute', 1440 and so on; return the service's code."""
    lookup = {
        "minute": "Minute",
        "1440": "Minute",
        "second": "Second",
        "86400": "Second",
    }
    key = str(cadence).strip().lower()
    if key not in lookup:
        raise ValueError(
            f"cadence must be one of {CADENCES} (or 1440 / 86400), got {cadence!r}"
        )
    return lookup[key]


def _as_date(value: date | datetime) -> date:
    return value.date() if isinstance(value, datetime) else value
