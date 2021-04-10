import glob
import math
from os import path, mkdir
import random

import cv2
import dlib
from imutils import face_utils
import numpy as np
import pandas as pd
from skimage.util import random_noise

DATASET_PATH = "C:\\youtube_faces"

class DetectorInput:
    """dlib input container class."""
    def __init__(self, file: str, top: int, left: int, width: int, height: int, points: list):
        self.file = file
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        self.points = points

def build_xml(inputs: list, filename: str):
    xmlstr = "<dataset>\n"
    xmlstr += "  <images>\n"
    for next_input in inputs:
        xmlstr += "    <image file='%s'>\n" % next_input.file
        xmlstr += "      <box top='%d' left='%d' width='%d' height='%d'>\n" % (next_input.top, next_input.left, next_input.width, next_input.height)
        for i, (x, y) in enumerate(next_input.points):
            xmlstr += "        <part name='%d' x='%d' y='%d' />\n" % (i, x, y)
        xmlstr += "      </box>\n"
        xmlstr += "    </image>\n"
    xmlstr += "  </images>\n"
    xmlstr += "</dataset>"

    with open(filename, "w") as f:
        f.write(xmlstr)

def main():
    video_df = pd.read_csv(DATASET_PATH + "\\youtube_faces_with_keypoints_full.csv")
    npz_files = glob.glob(DATASET_PATH + "\\youtube_faces_*\\*.npz")
    video_ids = [x.split('\\')[-1].split('.')[0] for x in npz_files]

    full_paths = {}
    for video_id, full_path in zip(video_ids, npz_files):
        full_paths[video_id] = full_path

    detector = dlib.get_frontal_face_detector()

    if not path.exists("./data"):
        mkdir("./data")

    inputs = []
    for video_id, video_path in full_paths.items():
        video_info = video_df.loc[video_df["videoID"] == video_id]
        frame_count = int(video_info["videoDuration"].values[0])

        video_file = np.load(video_path)
        images = video_file["colorImages"]
        landmarks = video_file["landmarks2D"]
        for i in range(frame_count):
            original_image = images[:,:,:,i]
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

            # Also save a copy with Gaussian noise. Gaussian noise most-closely
            # matches the noise apparent in low-quality webcam footage.
            image_gaussian = random_noise(original_image, mode="gaussian", mean=0, var=0.05, clip=True)
            image_gaussian = (255 * image_gaussian).astype(np.uint8)

            image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            faces = detector(image, 1)
            if len(faces) != 1:
                continue
            left, top, width, height = face_utils.rect_to_bb(faces[0])

            pts = landmarks[:,:,i]
            pts_x, pts_y = pts.T
            pts_min_x = np.min(pts_x)
            pts_max_x = np.max(pts_x)
            pts_min_y = np.min(pts_y)
            pts_max_y = np.max(pts_y)
            # Check if any of the keypoints are in the frame
            if pts_max_x < left or pts_min_x > left + width or pts_max_y < top or pts_min_y > top + height:
                continue

            filename = "data/%s_%d.png" % (video_id, i)
            if not path.isfile(filename):
                cv2.imwrite(filename, original_image)

            filename_gaussian = "data/%s_%d_gaussian.png" % (video_id, i)
            if not path.isfile(filename_gaussian):
                cv2.imwrite(filename_gaussian, image_gaussian)

            next_input = DetectorInput(filename, top, left, width, height, list(pts))
            next_input_gaussian = DetectorInput(filename_gaussian, top, left, width, height, list(pts))

            inputs += [next_input, next_input_gaussian]

    random.shuffle(inputs)
    split_index = math.floor(len(inputs) * 0.9) # 90% train, 10% test since our dataset is somewhat large
    train_inputs = inputs[0:split_index]
    test_inputs = inputs[split_index:]
    build_xml(train_inputs, "youtube_faces_train.xml")
    build_xml(test_inputs, "youtube_faces_test.xml")

if __name__ == "__main__":
    main()
