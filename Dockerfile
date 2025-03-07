# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04 AS build

ARG python_version=3.12
ARG user_id=1000
ARG group_id=1000

SHELL ["/bin/sh", "-exc"]

# System dependencies for build stage
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libvips-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    python${python_version}-dev \
    python${python_version}-venv && \
    rm -rf /var/lib/apt/lists/*

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /usr/local/bin/uv

# Configure environment variables
ENV UV_PYTHON="python${python_version}" \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONOPTIMIZE=1

# Create virtual environment
RUN python${python_version} -m venv /app/.venv

# Ensure the virtual environment is used
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock /_project/

# Install project dependencies into the virtual environment
RUN --mount=type=cache,destination=/root/.cache/uv \
    cd /_project && \
    uv venv /app/.venv && \
    uv sync

# Copy source code
COPY src /app/src

# Final stage
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04 AS final

ARG python_version=3.12
ARG user_id=1000
ARG group_id=1000

SHELL ["/bin/sh", "-exc"]

# Create app user and group
RUN <<EOF
# Check if group with GID exists
if getent group $group_id > /dev/null; then
    GROUP_NAME=$(getent group $group_id | cut -d: -f1)
else
    GROUP_NAME=app
    groupadd --gid $group_id $GROUP_NAME
fi

# Check if user with UID exists
if getent passwd $user_id > /dev/null; then
    USER_NAME=$(getent passwd $user_id | cut -d: -f1)
    usermod -d /app -g $GROUP_NAME $USER_NAME
else
    useradd --uid $user_id --gid $group_id --home-dir /app app
fi
EOF

# Runtime dependencies
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    wget \
    make \
    git \
    libvips \
    ffmpeg \
    libsm6 \
    libxext6 \
    python${python_version} \
    libcublas12 \
    libcusparse12 && \
    rm -rf /var/lib/apt/lists/*

# Environment configuration
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONOPTIMIZE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PYTHONPATH="/app/src"

COPY --from=build /usr/local/bin/uv /usr/local/bin/uv

# Create symbolic links for libvips
RUN ln -sf /lib/x86_64-linux-gnu/libvips.so.42 /usr/lib64/libvips.so.42
RUN ln -sf /lib/x86_64-linux-gnu/libvips-cpp.so.42 /usr/lib64/libvips-cpp.so.42

# Make sure to run ldconfig after creating symlinks
RUN ldconfig

# cuSPARSELt
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm -f /etc/apt/sources.list.d/cuda*.list && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get -y install libcusparselt0 libcusparselt-dev && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.1-1_all.deb

# Copy virtual environment and application files
COPY --chown=$user_id:$group_id --from=build /app /app
COPY --chown=$user_id:$group_id . /app

# Create necessary directories
RUN mkdir -p /app/data /app/checkpoints && \
    chown -R $user_id:$group_id /app

USER $user_id:$group_id
WORKDIR /app
CMD ["/bin/bash"]
EXPOSE 8080
