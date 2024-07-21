FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    gcc-9 \
    g++-9 \
    pip \
    python3-dev \
    python3 \
    wget \
    intel-mkl-full \
    libgoogle-glog-dev \
    protobuf-compiler \
    libprotobuf-dev \
    libfreeimage3 \
    libfreeimage-dev \
    xorg-dev \
    nvidia-driver-535 \
    unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY External External
RUN cd External && \
    wget -nv https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu116.zip -O libtorch.zip && \
    unzip -qq libtorch.zip -d . && \
    rm libtorch.zip

RUN wget -nv https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz -O cmake-dist.tar.gz && \
    tar xzf cmake-dist.tar.gz && \
    mv cmake-3.28.3-linux-x86_64 cmake-dist && \
    rm cmake-dist.tar.gz

ENV CC=gcc-9
ENV CXX=g++-9
ENV CUDAHOSTCXX=g++-9

COPY cmake cmake
COPY configs configs
COPY loss loss
COPY scenes scenes
COPY shader shader
COPY src src
COPY .clang-format .
COPY CMakeLists.txt .

RUN mkdir build && \
    cd build && \
    ../cmake-dist/bin/cmake -DCMAKE_PREFIX_PATH="./External/libtorch/;" \
## Quick fix to specify the CUDA architecture version as Docker cannot detect it during build stage
    -DSAIGA_CUDA_ARCH=8.9 -DTCNN_CUDA_ARCHITECTURES=8.9 -DTORCH_CUDA_ARCH_LIST=8.9 .. && \
    make && \
    make install

RUN mkdir /app/experiments
ENV PATH ${PATH}:/app/build/bin