# 1. Introduction
Spconv_lite is a simplied version for [spconv project](https://github.com/traveller59/spconv) by Yan Yan

The work in this repository is mainly involved with following papers:
* [3D Semantic Segmentation with Submanifold Sparse Convolutional Networks](https://arxiv.org/abs/1711.10275)
  * the sparse convolution and active sites concept was proposed.
*  [SECOND: Sparsely Embedded Convolutional Detection](https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf)
   * a significant performance improvement for sparse convolution was proposed. 
   * the author shared their excellent implementation, which named spconv.
* [Three-dimensional Backbone Network for 3D Object Detection in Traffic Scenes](https://arxiv.org/abs/1901.08373)
    * a excellent review with accurate math notation.


I made following changes:
* import and slightly modifed a subset of functions from spconv based on my own need. 
* refactor source codes
* fix compiler warning after analysing the source codes and algorihtm
* use docker container as building/deployment toolchain
* add my understanding about sparse convolution algorithm

# 2. Build
#### dependencies for usage
* python 3.8
* pytorch 1.6.0
* tqdm (for progressbar in unittest)
* numpy


#### build with docker container

pull the builder container

```bash
docker pull masszhou/toolchains:dev-spconv-lite-1.0
```

to build spconv_lite, under the root path of this project `{spconv_lite_root}/` run

```bash
docker build -t masszhou/spconv_lite:1.0 -f docker/build_spconv_lite.Dockerfile .
```

start a container by

```bash
docker run -d masszhou/spconv_lite:1.0
```

copy pip package to host, then install package in your virtualenv
```bash
docker cp <CONTAINER_ID>:/root/spconv_lite/dist/spconv_lite-1.0.0-cp38-cp38-linux_x86_64.whl .
pip install spconv_lite-1.0.0-cp38-cp38-linux_x86_64.whl
```

shutdown container

```bash
docker stop <CONTAINER_ID>
```

run unittest under `{spconv_lite_root}/unittest`

```bash
python -m test.test_all
```

# 3. My Understanding about Sparse Convolution

# 4. ToDo
* [ ] add examples
