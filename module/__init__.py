import cv2
import os
from os.path import join as opj
from PIL import Image
import numpy as np
import time
from .utils import get_root_logger
from .face_detector.retinaface.align_faces import warp_and_crop_face, get_reference_facial_points


def add_border(im, bordersize):
    row, col = im.shape[:2]
    bottom = im[row-2:row, 0:col]

    border = cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize,
                                left=bordersize,
                                right=bordersize,
                                borderType=cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
    return border


def face_alignment(img, facial5point, output_size):
    """rotate image based on facial5point

    args:
        img: cv2 image
        facial5point: 5point landmarks (support for retinaface and mtcnn)
        output_size: size image need to return
    return:
        dst_img: rotated image in cv2 image
    """

    facial5point = np.reshape(facial5point[0], (2, 5))

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    dst_img = warp_and_crop_face(img, facial5point,
                                 reference_pts=reference_5pts,
                                 crop_size=output_size)
    return dst_img


def initial_face_detector(conf, logger, device):
    """initial face detector

    args:
        conf: config for application
        logger: logging setting
        device: device setting
    return:
        face_detector: retinaface detector
    """
    from .face_detector.retinaface.detector import RetinafaceDetector
    face_detector = RetinafaceDetector(conf, device)
    logger.info('Retinaface detector loaded')
    return face_detector


def detect_faces(conf, logger, face_detector, original_image):
    image = original_image.copy()
    logger.info(f'using Retinaface')
    short_edge = min(image.shape[0], image.shape[1])
    scale_percent = 1
    image_copy = image.copy()
    faces = []
    if short_edge >= 500:
        scale_percent = float(500 / short_edge)
        width = int(image.shape[1] * scale_percent)
        height = int(image.shape[0] * scale_percent)
        dim = (width, height)
        image = cv2.resize(image, dim, cv2.INTER_AREA)
    logger.info(f'put image through Retina neural network')
    # frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    _, facial5points, frame_bgr, boxes = face_detector.detect_faces(image)
    if scale_percent != 1:
        boxes = np.multiply(boxes, (1/scale_percent))
        facial5points = np.multiply(facial5points, (1/scale_percent))

    # frame_bgr = cv2.cvtColor(np.array(image_copy), cv2.COLOR_RGB2BGR)
    frame_bgr = image_copy.copy()
    logger.info(f'{len(boxes)} face/faces detected')
    i = time.time()
    if len(facial5points) > 0 and len(boxes) > 0 and len(facial5points) == len(boxes):
        for facial5point in facial5points:
            i += 1
            facial5point = np.expand_dims(facial5point, axis=0)
            face = face_alignment(frame_bgr, facial5point,
                                  (conf.require_size, conf.require_size))
            face_ = Image.fromarray(face[..., ::-1])
            faces.append(face_)
            logger.info(f'infer through faces')
    return boxes, faces


def initial_log(conf):
    # mkdir_or_exist(osp.abspath(conf.work_dir))
    os.makedirs(conf.log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = opj(conf.log_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)

    return logger
