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


def load_module(conf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = module.initial_log(conf)
    face_detector = module.initial_face_detector(conf, logger, device)
    return face_detector, logger


if __name__ == '__main__':
    print('loading backbone')
    face_detector, logger = load_module(config)
    # convert detector
    face_detector.export_onnx()
    print('Export done')
