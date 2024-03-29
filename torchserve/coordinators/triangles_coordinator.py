from collections import defaultdict
import uuid
import time
import numpy as np
import base64
import cv2
import torch
from ts.torch_handler.base_handler import BaseHandler
from ilock import ILock
import warnings
from torchvision.transforms.functional import adjust_gamma

from methods.constants import (
    CUDA_ENABLED,
    TRIANGLE_MODEL_PATH, 
    MAP_LOCATION,
    VOD_TRIANGLE_PATH,
    VOD_TRIANGLE_BATCHES
)

from predictors.triangle.infer_standalone import infer, load_model
from predictors.vod_triangles.src.batch_inference import (
    infer_batch as vod_triangle_inference,
    fetch_model as vod_triangle_fetch_model
)
from methods.misc import (
    fetch_triangles_translators_batches,
    convert_coords_list2dicts,
    convert_yolo_output2dict,
    broaden_yolo_output,
    filter_yolo_output,
)
from ts.metrics.dimension import Dimension

class TriangleHandler(BaseHandler):
    
    def initialize(self, context):
        """
        Args:
            context: representing information about the available GPU

        Processes:
            initiates table detection model, cell detection model, easyocr model
        Returns:
            None
        """
        properties = context.system_properties
        self.context=context
        self.map_location = MAP_LOCATION
        self.device = MAP_LOCATION
        # assert self.device == "cuda", "GPU ISNT RECOGNIZED"
        print("DEVICE", self.device)
        self.triangle_model = load_model(TRIANGLE_MODEL_PATH, map_location=self.device)
        print(VOD_TRIANGLE_PATH)
        self.vod_triangle_model, self.vod_triangle_normalize = vod_triangle_fetch_model(ckpt=VOD_TRIANGLE_PATH, device=MAP_LOCATION)
        print("loaded yolo")
        self.triangle_count_dm = Dimension('level','trianglescount')
        # warm up triangle model
        if self.device != 'cpu':
            self.triangle_model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.triangle_model.parameters())))
        print("loaded vod_triangles")
        return self

    def preprocess(self, data):
        """
        Args:
            data: list of binary images
        Returns:
            list of cv2 images
        """
        
        images = []
        for row in data:
            image = row.get('data') or row.get('body')

            if isinstance(image, str):
                image = base64.b64decode(image)

            img = np.frombuffer(image, dtype=np.uint8).reshape((640, 640, 3))
            if img.shape != (640, 640, 3):
                warnings.warn(f"img.shape isnt (640,640,3)!={img.shape} undefined behaviour and wrong results can be returned")
          
            img = img.transpose((2, 0, 1))  # h, w, c --> c, h, w
            images.append(img)

        images = np.ascontiguousarray(np.array(images))
        images = torch.from_numpy(images)
        return images

    def inference_triangles(self, data, *args, **kwargs):
        """
        Args:
            data: list of cv2 images
        Returns:

        """
        tic = time.time()
        results = infer(data, model=self.triangle_model)
        
        # print(f"BatchSize.Batches:{len(data)}")        
        idx = str(uuid.uuid4())
        self.context.metrics.add_time(
            'YoloInferenceTriangleTimeForBatch',
            (time.time() - tic) * 1000,
            idx, 'ms'
        )

        return results

    def inference_vertices(self, imgs, triangles_bboxes, *args, **kwargs):
        """
        Args:
            data: list of cv2 images
        Returns:
            
        """

        tic = time.time()
        results = defaultdict(dict)
        for img_index in range(len(imgs)):
            results[img_index] = {
                "img_index": img_index,
                "triangles": []
            }  # has to be int because of sorting]
        triangles_cnt_ = 0
        for ind, (triangles, translators) in enumerate(fetch_triangles_translators_batches(
                yolo_output=triangles_bboxes, 
                imgs=imgs,
                n_batch=VOD_TRIANGLE_BATCHES,
                device=self.device
            )):
            tic_justinf = time.time()
            vertices = vod_triangle_inference(triangles, self.vod_triangle_model, self.vod_triangle_normalize, device=self.device)
            idx = str(uuid.uuid4())
            self.context.metrics.add_time(
                'TrianglesInternalInferenceTimeForBatch',
                (time.time() - tic_justinf) * 1000,
                idx, 'ms'
            )
            for translator, triangle_vertices in zip(translators, vertices):
                tr_vert = triangle_vertices.tolist()
                bbox = triangles_bboxes[translator.img_index][translator.triangle_index]
                vert_dicts = convert_coords_list2dicts(tr_vert, translator)
                bbox, confidence = convert_yolo_output2dict(bbox)
                img_dict = results[translator.img_index]
                    
                img_dict["triangles"].append(
                    {
                        "num" : translator.triangle_index,
                        "vertices" : vert_dicts,
                        "yolo_confidence" : confidence,
                        "bbox" : bbox
                    }
                )
        results = list((results.items()))
        results.sort(key=lambda a: a[0])
        idx = str(uuid.uuid4())
        self.context.metrics.add_time(
            'TrianglesInternalTotalTimeForBatch',
            (time.time() - tic) * 1000, 
            idx, 'ms'
        )
        self.context.metrics.add_counter('Stage2TotalTrianglesNumber', triangles_cnt_, idx)
        return list(map(lambda x: x[1], results))

    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.
        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.
        Returns:
            list : Returns a list of dictionary with the predicted response.
        """
        start_time = time.time()
        self.context = context
         
        metrics = self.context.metrics
        idx = str(uuid.uuid4())
        metrics.add_counter('BatchSize', len(data), idx)

        tic = time.time()
        data_preprocess = self.preprocess(data)
        metrics.add_time('PreprocessingTimeForBatch', (time.time() - tic) * 1000, idx,  'ms')
        tic = time.time()
        if CUDA_ENABLED:
            # with ILock("torchserve-mutex"): # dont use if models are slow
            data_preprocess = data_preprocess.to(self.device)
            data_preprocess = adjust_gamma(data_preprocess, 0.5)
            if not self._is_explain():
                yolo_output = self.inference_triangles(data_preprocess)
                # filter_yolo_output(yolo_output)
                broaden_yolo_output(yolo_output, 0.15)  # force yolo to get mode context to help stage2
                vod_vertices = self.inference_vertices(data_preprocess, triangles_bboxes=yolo_output)
            else:
                vod_vertices = self.explain_handle(data_preprocess, data)
            torch.cuda.empty_cache()
        else:
            data_preprocess = data_preprocess.to(self.device)
            data_preprocess = adjust_gamma(data_preprocess, 0.5)
            if not self._is_explain():
                yolo_output = self.inference_triangles(data_preprocess)
                # filter_yolo_output(yolo_output)
                broaden_yolo_output(yolo_output, 0.15)  # force yolo to get mode context to help stage2
                vod_vertices = self.inference_vertices(data_preprocess, triangles_bboxes=yolo_output)
            else:
                vod_vertices = self.explain_handle(data_preprocess, data)
            
        metrics.add_time('InferenceTotalTimeForBatch', (time.time() - tic) * 1000, idx, 'ms')

        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), idx, 'ms')
        
        return vod_vertices


if __name__ == '__main__':
    import os
    class Metrics:
            def add_time(self, *args, **kwargs):
                print(str(args))

            def add_metric(self, *args, **kwargs):
                print(str(args))
            def add_counter(self, *args, **kwargs):
                print(str(args))

    class Temp:
        def __init__(self, *args, **kwargs) -> None:
            self.system_properties = {"gpu_id": "0"}
            self.metrics = Metrics()
        def __str__(self):
            return ("here")

        def get_request_header(self, *args):
            return False
    print("tmp", Temp())

    # fname = "./predictors/triangle/fname_810.png"
    fname = "./predictors/triangle/fname_810.png"
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640)).astype(np.uint8)
    img_bytes = img.tobytes()
    open("../tests/triangles_heavy_img.png", "wb").write(img_bytes)

    tic = time.time()
    handler = TriangleHandler().initialize(Temp())
    #exit(0)
    print("Time for init", time.time() - tic)
    for _ in range(100):
        tic = time.time()
        results = handler.handle([{"data" : img_bytes} for _ in range(20)], Temp())
        toc = time.time()
        print("Handling Time", toc  - tic)
        #print(f"first results: {results[0]}")

    # for res in results[0]["triangles"]:
    #     import matplotlib.pyplot as plt

    #     xs, ys = [], []
    #     for vert in res["bbox"]:
    #         x_ = vert["x"]
    #         y_ = vert["y"]
    #         xs.append(x_)
    #         ys.append(y_)

    #     cv2.rectangle(img, [xs[0], ys[0]], [xs[1], ys[1]], (0, 255, 0), thickness=2)
    #     plt.imshow(img)
    #     xs, ys = [], []

    #     for vert in res["vertices"]:
    #         x_ = vert["x"]
    #         y_ = vert["y"]
    #         xs.append(x_)
    #         ys.append(y_)
    #     print(res)
    #     plt.scatter(xs,ys)
    # dirname = "deleteme"
    # if not os.path.isdir(dirname):
    #     os.mkdir(dirname)
    # plt.savefig(f"./{dirname}/giff.png")