# 运行环境

```
# CPU解码
docker pull shuairen/insightface:cuda11.1
# GPU解码
docker pull shuairen/insightface:cudacodec
# docker.sh为创建容器示例命令
```

# 生成人脸特征向量库

## 1. 生成包含人脸照片的人名文件夹

### 原始文件夹结构
```
照片文件夹
├── 张三.jpg
├── 李四.jpg
├── 王五-1.jpg
└── 王五-2.jpg
```

### 按照人名创建文件夹，并将照片移动到对应的人名文件夹
```
python3 create_name_folder.py <文件夹名称>
```

### 文件夹修改为如下结构

```
照片文件夹/
├── 张三/
│   └── 张三.jpg
├── 李四/
│   └── 李四.jpg
└── 王五/
    ├── 王五-1.jpg
    └── 王五-2.jpg
```

## 2. 构建人脸特征向量库

### 构建人脸特征向量库并保存人脸检测结果图
```
python3 tools/extract_features.py <文件夹名称>
```

### 生成如下结构文件夹，jpg是人脸检测结果，faiss_index.index是faiss向量库，metadata.pkl是人名字典

```
照片文件夹_result/
├── 张三.jpg
├── 李四.jpg
├── 王五-1.jpg
├── 王五-2.jpg
├── faiss_index.index
└── metadata.pkl
```

### 将faiss_index.index和metadata.pkl放入data文件夹中


# 运行命令

```
# CPU解码
python3 face.py --video <video> --json_port <json_port> --rtsp_port <rtsp_port> --dist-thres <dist-thres>
# GPU解码
python3 face_cudacodec.py --video <video> --json_port <json_port> --rtsp_port <rtsp_port> --dist-thres <dist-thres>
```