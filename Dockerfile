# CPU image (section 7).  Target: under 1.5 GB.
#
# torch is deliberately NOT installed here: the pixel pipeline runs on numpy and
# the CUDA path is optional, so the slim image stays small.  Use Dockerfile.gpu
# for NVDEC/NVENC and GPU gain application.
FROM python:3.12-slim-bookworm

ARG UID=1000
ARG GID=1000

# ffmpeg from Debian carries x264, x265, ProRes and FFV1, which is the full set
# section 7 asks for.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY uwnorm ./uwnorm

RUN pip install --no-cache-dir . \
    && python -c "import uwnorm, av, cv2, scipy, matplotlib; print(uwnorm.__version__)"

# Matplotlib needs a writable config dir; without this it warns on every run.
ENV MPLCONFIGDIR=/tmp/mpl \
    PYTHONUNBUFFERED=1

# Run as the host user so nothing lands in /work/out owned by root.
RUN groupadd -g ${GID} uwnorm 2>/dev/null || true \
    && useradd -m -u ${UID} -g ${GID} uwnorm 2>/dev/null || true \
    && mkdir -p /work/in /work/out /work/cache \
    && chown -R ${UID}:${GID} /work

USER ${UID}:${GID}
VOLUME ["/work/in", "/work/out", "/work/cache"]
WORKDIR /work

ENTRYPOINT ["uwnorm"]
CMD ["--help"]
