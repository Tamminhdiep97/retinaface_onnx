# retinaface_onnx

## This repo is based on [https://github.com/biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) which updated with the ability of converting model backbone into onnx model, allow using cpu for inferencing in realtime (TLDR: It's fast when running on cpu)<br />

### Setup
1. Using conda to create a virtual environment
```powershell
conda create --name py36 python=3.6
conda activate py36
```
2.  install needed python lib
```powershell
pip install -r requirements.txt
```
3.  Download weight file [here](https://drive.google.com/drive/folders/1_MCIVXaGfsC_ZacBtVyCIX9T21lai75H?usp=sharing) and put it in
```powershell
/retinaface_onnx/module/face_detector/retinaface/weights/
```
4. Run scripts

-   Change setting in file config.py according to your need
-   Script running **Realtime** detect face:
    ```powershell
    python infer.py
    ```
-   Script convert **pytorch** model into **onnx** model
    ```powershell
    python convert_onnx.py
    ```

#### Performence

On my laptop (CPU: AMD Ryzen 5 3500U, 12Gb Ram), the smallest backbone (MobileNet) archive performance of 30ms/image

### Reference:
- https://github.com/biubug6/Pytorch_Retinaface
- https://pytorch.org/docs/stable/onnx.html
- https://github.com/onnx/onnx/issues/2939
- https://github.com/pytorch/pytorch/blob/326d777e5384a621306330b5af0f2857843fe544/test/onnx/test_operators.py#L277