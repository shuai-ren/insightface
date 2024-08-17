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
COPY retinaface.py arcface_onnx.py /usr/local/lib/python3.8/dist-packages/insightface/model_zoo/

# opencv
RUN apt-get update && apt-get install --no-install-recommends -y  \
    git wget sudo build-essential vim cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev \ 
    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev \
    freeglut3-dev mesa-common-dev libgtkglext1 libgtkglext1-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    checkinstall yasm libv4l-dev libtbb-dev libqt4-dev libgtk2.0-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils \
    libswscale4 libswscale-dev libavresample3 libavresample-dev \
    python3.8-dev \
    ffmpeg libsm6 libxext6 && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* 

COPY opencv_cudacodec_build /opt/
RUN cd /opt/ && \
    cp cuviddec.h nvcuvid.h nvEncodeAPI.h /usr/include/ && \
    mv *.h /usr/local/cuda/include/ && \
    mv libnvcuvid.so.470.82.01 /usr/lib/x86_64-linux-gnu/ && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so.470.82.01 /usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so.470.82.01 /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 && \
    mv libnvidia-encode.so.470.82.01 /usr/lib/x86_64-linux-gnu/ && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.470.82.01 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so   

RUN cd /opt/opencv-4.7.0/ && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=DEBUG \
    -D CMAKE_INSTALL_PREFIX=/usr/local/opencv-4.7.0 \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=ON \
    -D BUILD_opencv_hdf=OFF \
    -D WITH_OPENGL=ON \
    -D WITH_OPENMP=ON \
    -D WITH_GTK=ON \
    -D WITH_VTK=ON \
    -D WITH_TBB=ON -D WITH_GSTREAMER=ON -D WITH_CUBLAS=ON -D WITH_GTK_2_X=ON \
    -D BUILD_EXAMPLES=ON -D WITH_FFMPEG=ON -D OPENCV_GENERATE_PKGCONFIG=ON  \
    -D OPENCV_ENABLE_NONFREE=ON -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_NVCUVID=ON \
    -D BUILD_opencv_cudacodec=ON  -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1  -D WITH_CUBLAS=1 \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-4.7.0/modules/ \
    -D HAVE_opencv_python3=ON -D PYTHON_EXECUTABLE=/usr/bin/python3.8 \
    -D PYTHON3_INCLUDE_DIR=/usr/include/python3.8 \
    -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so \
    -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
    -D BUILD_EXAMPLES=ON -D WITH_OPENGL=ON -D WITH_GTK_2_X=ON -D WITH_GTK=OFF \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.1 ..  && \
    make -j12 && make -j12 install


RUN echo "/usr/local/opencv-4.7.0/lib" >> /etc/ld.so.conf.d/opencv4.7.conf && ldconfig
ENV PKG_CONFIG_PATH=/usr/local/opencv-4.7.0/lib/pkgconfig
ENV LD_LIBRARY_PATH=/usr/local/opencv-4.7.0/lib:$LD_LIBRARY_PATH

RUN cp /opt/opencv-4.7.0/build/lib/python3/cv2.cpython-38-x86_64-linux-gnu.so /usr/local/lib/python3.8/dist-packages/

RUN pip uninstall opencv-python-headless -y
