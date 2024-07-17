FROM nvcr.io/nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-get update && apt-get install --no-install-recommends -y  \
    git wget sudo build-essential vim\
    python3 python3-setuptools python3-pip python3-dev python3-tk \
    ffmpeg libsm6 libxext6 && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* 

RUN apt-get update && apt-get install --no-install-recommends -y software-properties-common && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* 

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install --no-install-recommends -y  \
    python3.8 python3.8-dev && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* 

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2

RUN python3.8 -m pip install --upgrade pip

ENV LANG C.UTF-8

RUN ln -svf /usr/bin/python3.8 /usr/bin/python
RUN pip3 --no-cache-dir install flask faiss-cpu onnxruntime-gpu 
RUN pip3 install albumentations>=1.4.10
RUN pip3 install git+https://github.com/deepinsight/insightface.git#subdirectory=python-package
