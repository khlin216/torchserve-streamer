from collections import defaultdict
from distutils.dir_util import copy_tree
from unittest import result
from PIL import Image
import uuid
import time

import numpy as np
import mmcv
import cv2


from coordinator import TriangleHandler


if __name__ == "__main__":
    """
        check ../torchserve/coord.sh to run this test
        cp ../tests/test_coordinator_frames.py tst__.py
        cp coordinators/triangle_coordinator.py ./coordinator.py
        python tst__.py
        rm tst__.py
    """
    class Metrics:
        def add_time(self, *args, **kwargs):
            # print(str(args))
            pass
        def add_metric(self, *args, **kwargs):
            # print(str(args))
            pass
            
    class Temp:
        
        def __init__(self, *args, **kwargs) -> None:
            self.system_properties = {"gpu_id": "0"}    
            self.metrics = Metrics()

        def __str__(self):
            return ("here")

        def get_request_header(self, *args):
            return False

    print("tmp", Temp())
    handler = TriangleHandler().initialize(Temp())
    import os
    for file in os.listdir("../images"):
        file_ = os.path.join("../images", file)
        cvimg = cv2.imread(file_)
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        cvimg = cv2.resize(cvimg, (640, 640), interpolation=cv2.INTER_NEAREST)
        print(cvimg.shape)
        tic = time.time()
        
        results = handler.handle([{"data": cvimg.tobytes()}], Temp())
        pilim = Image.fromarray(cvimg)
        print(file_, results)
        for res in results[0]["triangles"]:
            from PIL import Image, ImageDraw

            draw = ImageDraw.Draw(pilim)
            
            xs, ys = [], []
            
            for vert in res["bbox"]:
                x_ = vert["x"]
                y_ = vert["y"]
                xs.append(x_)
                ys.append(y_)
            color = tuple((np.random.random(size=3) * 256).astype(int))
            print(color)
            print(res)
            draw.rectangle((xs[0], ys[0], xs[1], ys[1]), outline=color, width=2)
            draw.text((xs[0]+10, ys[0]+10), f"Conf={round(res['yolo_confidence'],2)}", fill="blue")
            xs, ys = [], []
            for vert in res["vertices"]:
                x, y = vert["x"], vert["y"]
                draw.ellipse((x-3, y-3, x+3, y+3), "green", width=5)

        os.makedirs("./deleteme/frames/",exist_ok=True)
        pilim.save(f"./deleteme/{file}")
