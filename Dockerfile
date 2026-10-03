# MuViSeg runtime image.
#
# Justified by the dependency stack rather than by convention: the project is
# pinned to CUDA 12.8 PyTorch wheels, and the downstream navigation experiment
# needs habitat-sim, which is painful to install natively.
#
#   docker build -t muviseg .
#   docker run --gpus all -it -v /path/to/datasets:/datasets muviseg
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH

# libgl1 / libglib2.0-0 are needed by opencv-python; git is needed by
# setup_third_party.sh.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

# Dependencies first, so edits to the source do not invalidate the layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra eval --extra dev

COPY . .
RUN uv sync --frozen --extra eval --extra dev

# Backbone weights and datasets are deliberately NOT baked in: they are large and
# come from third parties under their own terms. Mount them, then run
#   bash setup_third_party.sh
CMD ["/bin/bash"]
