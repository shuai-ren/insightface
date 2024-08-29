import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = '1'
import argparse
import datetime
import time
import pickle
from collections import deque
from PIL import Image, ImageDraw, ImageFont

import cv2
import faiss
import numpy as np
from insightface.app import FaceAnalysis
from stream import start


def load_index_and_metadata(index_file, metadata_file):
    index = faiss.read_index(index_file)
    with open(metadata_file, 'rb') as f:
        features_dict = pickle.load(f)
    return index, features_dict


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def main():
    app = FaceAnalysis(name='buffalo_l', root='./', allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=args.conf_thres)

    # warmup
    app.get(np.random.randn(640, 640, 3))

    index, features_dict = load_index_and_metadata(args.index_file, args.metadata_file)
    font = ImageFont.truetype(args.font_file, 30)

    data_deque = deque(maxlen=100)
    start(data_deque, args.json_port, args.rtsp_port, args.mjpeg_port, out_size=None, fps=25, max_length=50, timeout=10)

    param = cv2.cudacodec.VideoReaderInitParams()
    param.udpSource = True
    video_reader = cv2.cudacodec.createVideoReader(args.video, params=param)
    stream = cv2.cuda_Stream()

    frame_id = 0
    reconnected = False
    objects = []
    color = (0, 0, 255)

    start_time = time.time()
    while True:
        t0 = time.time()
        ret, frame_gpu = video_reader.nextFrame()
        t1 = time.time()
        if not ret or frame_gpu is None:
            if reconnected:
                break
            video_reader = cv2.cudacodec.createVideoReader(args.video, params=param)
            reconnected = True
            continue
        
        frame_bgr_gpu = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGRA2BGR, stream=stream)
        frame_bgr_gpu_outsize = cv2.cuda.resize(frame_bgr_gpu, (out_width, out_height), stream=stream)
        frame = frame_bgr_gpu_outsize.download(stream=stream)
        orig_w, orig_h = frame_bgr_gpu.size()

        t2 = time.time()
        frame_id += 1

        if frame_id % args.interval == 0:
            objects = []

            new_width = 640
            new_height = int(new_width / out_width * out_height)
            resized_frame = cv2.cuda.resize(frame_bgr_gpu, (new_width, new_height), stream=stream).download(stream=stream)

            t3 = time.time()
            faces = app.get(resized_frame)
            t4 = time.time()

            for face in faces:
                normalized_features = normalize_vector(face["embedding"])
                conf = face["det_score"].tolist()
                bbox = face.bbox / (new_width, new_height, new_width, new_height)
                bbox_orig = (bbox * (orig_w, orig_h, orig_w, orig_h)).astype(int).tolist()
                bbox_draw = (bbox * (out_width, out_height, out_width, out_height)).astype(int)
                distances, indices = index.search(np.array([normalized_features]), 1)
                person_name = "未知"
 
                distance = distances[0][0].tolist()
                if distance < args.dist_thres:
                    for n, i in features_dict.items():
                        if indices[0][0] in i:
                            person_name = n
                            break

                obj = {
                    "name": person_name,
                    "distance": distance,
                    "confidence": conf,
                    "face_coordinates": bbox_orig
                }

                objects.append(obj)

                image_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(image_pil)
                text = "{}: {:.1f}% {:.2f}".format(person_name, conf*100, distance)
                draw.text((bbox_draw[0], bbox_draw[1] - 30), text, font=font, fill=color)
                frame = np.array(image_pil)
                cv2.rectangle(frame, (bbox_draw[0], bbox_draw[1]), (bbox_draw[2], bbox_draw[3]), color, 2)

            t5 = time.time()
            print('read: %.3f, download: %.3f, \tprep: %.3f, \tinfer: %.3f, \tpost: %.3f' % (1/(t1-t0), 1/(t2-t1), 1/(t3-t2), 1/(t4-t3), 1/(t5-t4)))
        print('fps: %.3f, \tavg_fps: %.3f, \tbuffer: %g' % (1 / (time.time() - t0), frame_id / (time.time() - start_time), len(data_deque)))

        json = {"frame_id": frame_id,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "objects": objects
        }
        data_deque.append((json, frame))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='video url')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--dist-thres', type=float, default=1.0, help='distance threshold')
    parser.add_argument('--interval', type=int, default=5, help='sample interval')
    parser.add_argument('--json_port', type=int, help='json port')
    parser.add_argument('--rtsp_port', type=int, help='rtsp port')
    parser.add_argument('--mjpeg_port', type=int, help='mjpeg port')
    parser.add_argument('--index-file', type=str, default="./data/faiss_index.index", help='index file')
    parser.add_argument('--metadata-file', type=str, default="./data/metadata.pkl", help='metadata file')
    parser.add_argument('--font-file', type=str, default="./font/simsun.ttc", help='font file')

    args = parser.parse_args()
    print(args)

    import json

    cwd = os.getcwd()[7:]
    business_json_list = ["business.json", "/root/MBAB/AI/{}/etc/business.json".format(cwd)]
    out_width = 1280
    out_height = 720
    add_line = lines = color = width = None
    for business_json in business_json_list:
        if os.path.isfile(business_json):
            business = json.load(open(business_json))
            params = business["business_params"]
            out_size = params.get("out_size")
            if out_size is not None:
                out_width = out_size.get("out_width")
                out_height = out_size.get("out_height")
                
            add_line = params.get("add_line")
            if add_line is not None:
                lines = add_line.get("lines")
                width = add_line.get("width")
                color = add_line.get("color")
            break

    main()
