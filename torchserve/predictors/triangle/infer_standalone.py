import torch
import numpy as np
import os
import sys

yolodir = os.path.abspath("./predictors/yolov7")
sys.path.append(yolodir)
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.general import check_img_size
sys.path.remove(yolodir)



def load_model(weights, map_location):
    
    model = attempt_load(weights, map_location=map_location)
    
    return model.to(map_location)

def infer(imgs: np.ndarray, weights=None, model=None, device="cpu", imgsz=640, half: bool = False,
          conf_thres=0.25, iou_thres=0.45, classes=None):
    """
    Usage:

    img = cv2.imread("img_3-4_19_9.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr --> rgb
    img = cv2.resize(img, (640, 640))
    img = img.transpose((2, 0, 1))  # h, w, c --> c, h, w
    results = infer_standalone.infer(weights="last.pt", device="cpu", img=img,)
    print(results)

    [tensor([[3.50291e+02, 4.80507e+02, 4.48374e+02, 5.88482e+02, 9.51800e-01, 0.00000e+00],
            [5.25183e+02, 4.86838e+02, 5.90338e+02, 5.75566e+02, 9.50960e-01, 0.00000e+00],
            [1.97287e+02, 7.56277e+01, 2.82298e+02, 1.43393e+02, 9.29669e-01, 0.00000e+00],
            [3.63544e+02, 3.09668e+01, 4.12942e+02, 1.76267e+02, 9.25530e-01, 0.00000e+00],
            [1.66383e+02, 2.73124e+02, 3.12851e+02, 3.89444e+02, 9.11006e-01, 0.00000e+00],
            [2.69657e+01, 2.93518e-01, 9.89929e+01, 2.04104e+02, 9.05363e-01, 0.00000e+00],
            [6.81570e+00, 2.99020e+02, 1.59248e+02, 3.76828e+02, 8.69054e-01, 0.00000e+00],
            [1.81473e+02, 4.63728e+02, 2.98181e+02, 6.00940e+02, 8.66575e-01, 0.00000e+00]])]
"""
    
    # load assets -- this should probably go into initialize()
      # load FP32 model
    if weights is not None and model is not None:
        raise("You should pass either weights or model")
    if weights is not None:
        model = load_model(weights, map_location=device)

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # do inference -- this should probably go into inference()
    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs)
        imgs = imgs.to(device)
    half = False
    imgs = imgs.half() if half else imgs.float()  # uint8 to fp16/32
    imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
    if imgs.ndimension() == 3:  # add pseudo-batch dimension
        imgs = imgs.unsqueeze(0)
    with torch.no_grad():
        pred = model(imgs, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes)
        
        pred = [p.detach().cpu().numpy().astype(np.float32).tolist()  for p in pred]
        
        return pred

