import math

import cv2
import dlib
from imutils import face_utils
import numpy as np

from eye_centers import get_centre, EYE_SIZE
from eye_utils import get_gradient, weight_preprocess, normalize_gradient, eye_pad_min_x, eye_pad_max_x, eye_pad_min_y, eye_pad_max_y

TARGET_INPUT_SIZE = (256, 256)

right_eye_points = [30, 31, 32, 33, 35, 36]
left_eye_points = [37, 38, 39, 40, 41, 42]

def get_points_at_indices(arr, indices) -> list:
    return [arr[i] for i in indices]

def main():
    print("Loading models...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("youtube_faces_68_points.dat")
    print("Done!")

    device = cv2.VideoCapture(0)

    while device.isOpened():
        ret, image = device.read()
        if ret is None:
            break

        face_initial_height, face_initial_width = image.shape[:2]
        face_scale_x = face_initial_width / TARGET_INPUT_SIZE[0]
        face_scale_y = face_initial_height / TARGET_INPUT_SIZE[1]
        face_input = cv2.resize(image, TARGET_INPUT_SIZE)
        gray = cv2.cvtColor(face_input, cv2.COLOR_BGR2GRAY)

        faces = detector(gray, 1)

        for _, rect in enumerate(faces):
            points = predictor(gray, rect)
            points = face_utils.shape_to_np(points)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            #x, y, w, h = face_utils.rect_to_bb(rect)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            #for x, y in points:
            #    cv2.circle(image, (math.floor(x * face_scale_x), math.floor(y * face_scale_y)), 1, (0, 0, 255), -1)

            left_eye_pts = get_points_at_indices(points, left_eye_points)
            right_eye_pts = get_points_at_indices(points, right_eye_points)
            for _, eye_pts in enumerate([left_eye_pts, right_eye_pts]):
                # get separate bounding boxes around the eyes
                ex, ey = np.transpose(eye_pts)
                x1, y1, x2, y2 = (eye_pad_min_x(ex), eye_pad_min_y(ey), eye_pad_max_x(ex), eye_pad_max_y(ey))

                # calculate eye centers
                eye = gray[y1:y2, x1:x2]
                initial_height, initial_width = eye.shape[:2]
                eye_scale_x = initial_width / EYE_SIZE[0]
                eye_scale_y = initial_height / EYE_SIZE[1]

                eye = cv2.resize(eye, EYE_SIZE)
                grad = get_gradient(eye)
                grad = normalize_gradient(grad)

                weights = weight_preprocess(eye) / 255

                center = get_centre(weights, grad)

                center_x = int(math.floor(center[0] * eye_scale_x))
                center_y = int(math.floor(center[1] * eye_scale_y))

                # draw the center on the eye
                cv2.circle(image, (math.floor((x1 + center_x) * face_scale_x), math.floor((y1 + center_y) * face_scale_y)), 1, (0, 255, 0), 1)
        cv2.imshow("Output", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

if __name__ == "__main__":
    main()
