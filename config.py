log_dir = 'log_dir/'
detector_root = './module/face_detector/'

# retina detector
retinaface = True
require_size = 112
embedding_size = 512
detector_onnx = True

if retinaface:
    network = 'mnet'
    # network = 'resnet50'
    if network == 'resnet50':
        weights = detector_root + 'retinaface/weights/Resnet50_Final.pth'
        detector_backbone = detector_root + 'retinaface/weights/resnet50-19c8e357.pth'
        onnx_weights = detector_root + 'retinaface/weights/detector_resnet50.onnx'
    else:
        detector_backbone = detector_root + 'retinaface/weights/mobilenetV1X0.25_pretrain.tar'
        weights = detector_root + 'retinaface/weights/mobilenet0.25_Final.pth'
        onnx_weights = detector_root + 'retinaface/weights/detector_mobilenet.onnx'

    vis_thres = 0.8


# Tranformer <=> Normalize input ==============================================
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]



