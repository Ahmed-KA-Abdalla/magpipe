# magpipe — design note

Version 0.1, August 2026.

## 1. Purpose and scope

magpipe retrieves ground magnetometer observatory data from the
INTERMAGNET network, checks its quality, and produces figures suitable
for inspection by an analyst. It exists to demonstrate the shape of a
scientific data pipeline: retrieval from an external service, parsing of
a fixed-format community exchange file, quality control against stated
tolerances, and visualisation.

In scope:

- retrieval from the INTERMAGNET GIN web service, with caching;
- parsing of the IAGA-2002 exchange format;
- six quality checks with configurable tolerances;
- three figure types.

Out of scope for this version:

- persistent storage; frames are held in memory for the duration of a
  run, and cached files serve as the only durable artefact;
- scheduling and orchestration;
- any processing that alters the data, such as gap filling, despiking or
  baseline correction. The pipeline reports defects and leaves the
  values as the provider sent them.

## 2. Data source

Data are served by the Edinburgh Geomagnetic Information Node, operated
by the British Geological Survey, at
`https://imag-data.bgs.ac.uk/GIN_V1/GINServices`. Requests are
controlled entirely by URL query parameters.

Three properties of that interface drove design decisions.

**Parameter naming is inconsistent.** `Request` and `Format` are
capitalised while the remaining parameters begin with a lower-case
letter, and dates are plain `yyyy-mm-dd` rather than ISO 8601
timestamps. A malformed request returns HTTP 400 with no indication of
which parameter is at fault, so URL construction is confined to one
place, `DataRequest.params`, and covered by tests that assert the exact
parameter spelling.

**The interval may be given as an end date or as a duration in days.**
The service's default end date is one day after the start, which makes
the boundary behaviour ambiguous. The client sends `dataDuration`, which
is not ambiguous.

**Component orientation is selectable, and the choice is significant.**
Under `XYZF` or `HDZF` the service computes the total field from the
vector components. Under `XYZS` or `HDZS` it takes the total field from
an independent scalar instrument. This determines whether the
scalar/vector consistency check is a real comparison or an identity; see
section 5.

INTERMAGNET data are licensed CC-BY-NC and are subject to INTERMAGNET's
conditions of use. Downloads are cached under `data/`, which is excluded
from version control. The repository commits only six-hour extracts
under `tests/data/`, carrying the acknowledgement as comment records.

## 3. Architecture

Four modules, arranged so that each depends only on those above it.

    fetch     HTTP retrieval and on-disk caching
    parse     IAGA-2002 text -> DataFrame
    validate  DataFrame -> quality report
    plot      DataFrame -> figures

`parse`, `validate` and `plot` perform no network access and no
mutation of their inputs. `fetch` performs no parsing: it verifies that
a response resembles IAGA-2002 and writes it to the cache, and nothing
more.

This separation is the main design decision in the project, and it was
made to serve testability. A parser that fetches cannot be tested
without a network; a validator that reads files cannot be tested without
fixtures on disk. Because each stage takes data and returns data, the
test suite exercises 97% of the package without a network connection, a
database, or a display.

Two scripts compose the modules into workflows: `scripts/fetch_data.py`
downloads and optionally validates; `scripts/plot_day.py` reads the
cache and writes figures. Neither contains logic that is not
composition, so neither needs its own tests.

## 4. Parsing

The IAGA-2002 format is specified in Appendix E3 of the INTERMAGNET
Technical Reference Manual: twelve mandatory header records of fixed
width, optional comment records, one data header record, then one record
per sample.

Three decisions in the parser are worth stating.

**Header labels are matched case-insensitively with whitespace
collapsed.** Producers differ on capitalisation, and a parser that
demands one spelling rejects valid files. Whichever form arrives is
stored under a canonical name so callers have one key to use.

**Only three headers are treated as required** — `Format`, `IAGA CODE`
and `Reported` — because those are the three the module reads. The other
nine specification headers are reported as a warning when absent. A file
missing `Digital Sampling` is still perfectly readable, and rejecting it
would discard usable data to no purpose. When a required header is
absent the error lists the labels that were present, because the usual
cause is a spelling difference rather than an omission and that cannot
be diagnosed from the absence alone.

**Absent-value sentinels are counted before they are discarded.** The
format encodes absent values numerically: 99999.00 for a missing value
and 88888.00 for a value that was not observed. Both become NaN, but
the counts are retained separately, because the two mean different
things operationally. A missing value indicates a fault in an instrument
that should be reporting; a value that was never observed indicates an
instrument that does not report that component at all. Conflating them
would report a three-component observatory as permanently faulty.

Absent *samples* are a separate matter from absent *values*. A sample
omitted from the file leaves no row, so the parser reports what the file
contains and gap detection belongs to the validator.

## 5. Validation

Six checks, each a pure function of a frame and a `Thresholds` instance,
returning a list of `Issue`. No check raises on bad data. A pipeline
that aborts on the first bad sample cannot process an archive, so
defects are reported rather than thrown, and the caller decides what an
acceptable file is.

Severity separates two cases. An **error** means the file is unusable
downstream. A **warning** means it is degraded but serviceable. That
distinction is what makes a report actionable rather than merely
informative.

| Check | Detects | Severity |
| --- | --- | --- |
| `ordering` | duplicated or out-of-order timestamps | error |
| `cadence` | samples absent from a regular series | error |
| `completeness` | absent values beyond tolerance | error or warning |
| `range` | values outside plausible surface field limits | error |
| `spikes` | sample-to-sample changes too large to be geophysical | warning |
| `field_consistency` | disagreement between scalar and vector total field | error or warning |

Gaps are reported as intervals rather than as individual timestamps: a
three-hour outage is one finding, not 180.

### The scalar/vector consistency check

This is the only check that can detect a genuine instrument fault rather
than a file defect, and it required measurement rather than assumption.

The total field F is measured by a separate scalar magnetometer, so
comparing it against the magnitude computed from the vector components
tests two instruments against each other. An initial tolerance of 1.0 nT
on individual samples proved far too tight. Measured on definitive
one-minute data from Eskdalemuir and Hartland for 10 May 2024:

| Observatory | n | median | s.d. | range | 99th percentile of \|residual\| |
| --- | --- | --- | --- | --- | --- |
| ESK | 1440 | +0.13 nT | 0.52 nT | −4.40 to +5.30 nT | 2.60 nT |
| HAD | 1440 | +0.19 nT | 0.44 nT | −2.10 to +2.10 nT | 0.80 nT |

Those figures describe two distinct failure modes, so the check was
split accordingly. A **systematic offset**, taken as the median
residual, indicates a baseline error and is an error at 1.0 nT.
**Individual excursions** are ordinary instrument disagreement; they are
reported only when more than 1% of samples exceed 5.0 nT, and then as a
warning. Both observatories pass at these settings.

One caveat governs the interpretation of any report. Under an F
orientation the service derives the total field from the vector
components, so the residual is identically zero and the check passes
trivially. It is informative only under an S orientation. The default
thresholds were measured from `XYZS` data.

## 6. Visualisation

Three figures: per-element time series, the scalar/vector residual, and
one element from several observatories overlaid.

**Baselines are subtracted by default.** The components sit at tens of
thousands of nanotesla while the variation of interest is tens to
hundreds, so plotting absolute values renders the signal as a flat line.
The median over the plotted interval is removed and the offset is
recorded in the axis label, so nothing is concealed.

**Gaps are shaded rather than interpolated across.** Matplotlib joins
the points either side of an absent sample with a straight line, which
draws an outage as a smooth trend. Absent intervals are located by the
cadence check and marked.

No function calls `show` or `savefig`; each returns a figure, and the
caller decides what happens to it. This is what allows the tests to
inspect figures directly.

## 7. Deployment

The image is built in two stages so that pip and build tooling do not
reach the runtime layer, and runs as an unprivileged user. Two
environment variables are set because matplotlib otherwise attempts to
build a font cache in a home directory it cannot write to, which is a
common containerisation failure: `MPLBACKEND=Agg` and a writable
`MPLCONFIGDIR`.

`docker-compose.yml` defines two services over one image: `test` runs
the suite, `cli` runs any pipeline command with `data/` and
`docs/figures/` bind-mounted, so downloads and figures persist on the
host and INTERMAGNET data never enter the image.

Continuous integration runs on every push and pull request: ruff lint
and format checks; the suite on Python 3.11, 3.12 and 3.13 with
coverage; then a Docker build which runs the suite again inside the
image. The last of these verifies that the image is functional rather
than merely buildable.

## 8. Known limitations

- No persistent storage. Reprocessing an interval means reparsing the
  cached file.
- The cache has no expiry or size limit and is never pruned.
- Retrieval is sequential; a request for many observatories takes
  proportionally longer.
- Range thresholds are set for mid-latitude surface observatories and
  would reject valid data from high-latitude or equatorial sites.
- The spike threshold of 500 nT per minute is a plausibility limit
  rather than a measured one. Unlike the consistency tolerances it has
  not been calibrated against observed data.
- HDZ to XYZ conversion is implemented but not exercised against real
  files, since the observatories used report XYZ natively under the
  chosen orientation.
- One-second data is supported by the client but untested; at 86,400
  samples per day per observatory the memory behaviour of the current
  in-memory design has not been examined.

## 9. Possible extensions

Storage in PostgreSQL with an ingest-run table, so that reprocessing is
incremental and validation findings are queryable over time. Parallel
retrieval. Calibration of the spike threshold against a storm catalogue.
A comparison of the pipeline's cadence and completeness findings against
INTERMAGNET's own published data-availability statistics, which would
serve as an independent check on the validator itself.
