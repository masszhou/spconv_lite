# info of base image:
#     nvidia/cuda, 10.2-cudnn7-devel-ubuntu18.04
#
# Build:
#     docker build -t masszhou/spconv_lite:1.0 -f docker/build_spconv_lite.Dockerfile .
#
# Inspection:
#     docker run --gpus all -ti masszhou/spconv_lite:1.0 /bin/bash
#
# Notes:
#    1. ros-melodic-desktop has opencv 3.2 included
#

FROM masszhou/toolchains:dev-spconv-lite-1.0
LABEL maintainer="Zhiliang Zhou <zhouzhiliang@gmail.com>"
USER root
WORKDIR /root

RUN mkdir spconv_lite

COPY CMakeLists.txt /root/spconv_lite/
COPY setup.py /root/spconv_lite/
COPY components /root/spconv_lite/components/
COPY extern /root/spconv_lite/extern/
COPY spconv_lite /root/spconv_lite/spconv_lite/
COPY unittest /root/spconv_lite/unittest/

RUN cd spconv_lite \
    && python3.8 setup.py bdist_wheel

# test package need GPU container
# RUN pip install /root/spconv_lite/dist/spconv_lite-1.0.0-cp38-cp38-linux_x86_64.whl \
#     && cd /root/spconv_lite/unittest \
#     && python3.8 -m test.test_all
