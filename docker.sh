docker run -it --rm --gpus=all --net=host -v `find /usr/lib/x86_64-linux-gnu/ -name libnvcuvid.so.*.*`:/usr/lib/x86_64-linux-gnu/libnvcuvid.so.470.82.01 -v $PWD:/home -w /home insightface:cudacodec
