import config
import time
from random import randint
import os
from os.path import join as opj

import cv2
import torch
from PIL import Image
import numpy as np

import module


def draw_box_name(bbox, frame):
    color, thickness, r, d = (200, 200, 20), 4, 4, 15
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    # Top left
    cv2.line(frame, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(frame, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(frame, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(frame, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(frame, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(frame, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(frame, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(frame, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    return frame

def load_module(conf):
    if conf.detector_onnx:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = module.initial_log(conf)
    face_detector = module.initial_face_detector(conf, logger, device)
    return face_detector, logger


def detect(detector, image, conf, logger):
    boxes, faces = module.detect_faces(conf, logger, face_detector, image)
    if len(faces) == 0:
        return image
    for bbox in boxes:
        image = draw_box_name(bbox, image)
    return image


if __name__ == '__main__':
    print('loading')
    window_name = 'RETINAFACE_' + config.network + '_'
    if config.detector_onnx:
        window_name += 'ONNX'
    else:
        window_name += 'PYTORCH'
    face_detector, logger = load_module(config)

    vid = cv2.VideoCapture(0)
    while(True):
        ret, frame = vid.read()
        # frame = cv2.imread(opj('data_test', 'image_test2.jpg'))
        # ret = 1
        if ret:
            h, w, c = frame.shape
            resized_down = cv2.resize(frame, (int(w/2), int(h/2)),
                                      interpolation=cv2.INTER_LINEAR)
            result_image = detect(face_detector, resized_down,
                                     config, logger)
            cv2.imshow(window_name.upper(), result_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    vid.release()
    cv2.destroyAllWindows()
