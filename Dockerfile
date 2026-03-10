# Start from a CUDA-compatible base image
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Install miniconda
RUN apt-get update && apt-get install -y wget git && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Copy and create the conda environment
COPY hgrnenv2.yml .
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f hgrnenv2.yml

# Make the environment active by default
ENV PATH="/opt/conda/envs/hgrnenv/bin:$PATH"
SHELL ["conda", "run", "-n", "hgrnenv", "/bin/bash", "-c"]

# Copy and install DeepHCD
COPY . /DeepHCD
WORKDIR /DeepHCD
RUN pip install -e .
