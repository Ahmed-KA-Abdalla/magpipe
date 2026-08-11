# Multi-stage build. The builder installs dependencies into a virtual
# environment; the runtime stage copies only that environment and the
# source, so build tools do not reach the final image.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the metadata first so that the dependency layer is cached
# and reinstalled only when the dependencies themselves change, not on
# every source edit.
COPY pyproject.toml README.md ./
COPY magpipe/__init__.py magpipe/__init__.py
RUN pip install --no-cache-dir ".[dev]"

FROM python:3.12-slim AS runtime

# Matplotlib writes a font cache at import; give it a writable location
# and select a backend that needs no display.
ENV PATH="/opt/venv/bin:$PATH" \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Run as an unprivileged user. The container writes only to /app/data
# and /tmp, both of which this user owns.
RUN useradd --create-home --uid 1000 magpipe
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=magpipe:magpipe magpipe/ magpipe/
COPY --chown=magpipe:magpipe scripts/ scripts/
COPY --chown=magpipe:magpipe tests/ tests/
COPY --chown=magpipe:magpipe pyproject.toml README.md ./

RUN mkdir -p /app/data /app/docs/figures && chown -R magpipe:magpipe /app
USER magpipe

# A failing import is the most common way a built image is broken, and
# it is cheap to detect.
HEALTHCHECK --interval=30s --timeout=5s --retries=2 \
    CMD python -c "import magpipe.parse, magpipe.plot" || exit 1

ENTRYPOINT ["python"]
CMD ["-m", "pytest"]
