import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = '1'
import sys
import pickle

import cv2
import faiss
import numpy as np
from insightface.app import FaceAnalysis


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def extract_features_from_images(image_folder, result_folder):
    app = FaceAnalysis(name='buffalo_l', root='./', allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    features_dict = {}
    index = faiss.IndexFlatL2(512)
    i = 0
    for person_name in os.listdir(image_folder):
        person_folder = os.path.join(image_folder, person_name)
        if os.path.isdir(person_folder):
            features_dict[person_name] = []
            for image_file in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_file)
                image = cv2.imread(image_path)
                features = app.get(image)
                result = app.draw_on(image, features)
                cv2.imwrite(os.path.join(result_folder, image_file), result)
                normalized_features = normalize_vector(features[0]["embedding"])
                index.add(np.array([normalized_features]))
                features_dict[person_name].append(i)
                i += 1
    return features_dict, index


def save_index_and_metadata(index, features_dict, index_file, metadata_file):
    faiss.write_index(index, index_file)
    with open(metadata_file, 'wb') as f:
        pickle.dump(features_dict, f)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 extract_features.py <image_folder>")
        exit(1)

    image_folder = sys.argv[1]
    result_folder = os.path.abspath(image_folder) + "_result"
    os.makedirs(result_folder, exist_ok=True)
    index_file = os.path.join(result_folder, 'faiss_index.index')
    metadata_file = os.path.join(result_folder, 'metadata.pkl')

    features_dict, index = extract_features_from_images(image_folder, result_folder)
    save_index_and_metadata(index, features_dict, index_file, metadata_file)
