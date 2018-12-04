FROM python:3.6.3-jessie

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /stage

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy select files needed for installing requirements.
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
RUN pip install git+https://github.com/allenai/allennlp.git@0c532d50ac52bca48bf9906864daf6735d5d02de#egg=allennlp
RUN pip install cython


COPY bin/ bin/
COPY common/ common/
COPY dataset_readers/ dataset_readers/
COPY models/ models/
COPY modules/ modules/
COPY training_config/ training_config/
COPY __init__.py

# Compile Cython extension and install package.
RUN make build

CMD ["/bin/bash"]