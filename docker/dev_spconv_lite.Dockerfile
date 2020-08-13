# info of base image:
#     nvidia/cuda, 10.2-cudnn7-devel-ubuntu18.04
#
# Build:
#     docker build -t masszhou/toolchains:dev-spconv-lite-1.0 -f docker/dev_spconv_lite.Dockerfile .
#
# Inspection:
#     docker run --gpus all -ti masszhou/toolchains:dev-spconv-lite-1.0 /bin/bash
#
# Launch node:


FROM masszhou/toolchains:dev-cuda-10.2
LABEL maintainer="Zhiliang Zhou <zhouzhiliang@gmail.com>"

# install libtorch dependencies
RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install \
    libboost-all-dev \
    python3.8 \
    python3.8-dev \
    python3-yaml \
    python3.8-distutils \
    libomp-dev

RUN python3.8 -m pip install --upgrade pip setuptools wheel
RUN pip install numpy \
    scipy \
    matplotlib \
    pillow \
    torch \
    torchvision \
    scikit-image \
    tqdm \
    fire \
    opencv-python==4.2.0.32