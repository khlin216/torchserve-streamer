from collections import defaultdict
from distutils.dir_util import copy_tree
from unittest import result
from PIL import Image
import uuid
import time

import numpy as np
import base64
import mmcv
import cv2
import torch
from ts.torch_handler.base_handler import BaseHandler
from ilock import ILock

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
    convert_yolo_output2dict
)

class TraingleHandler(BaseHandler):
    
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
        self.vod_triangle_models = vod_triangle_fetch_model(ckpt=VOD_TRIANGLE_PATH, device=MAP_LOCATION)
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
            # self.context.metrics.add_metric('PackageSize', len(image) // 8000, 'KB', str(uuid.uuid4()))
            if isinstance(image, str):
                image = base64.b64decode(image)

            img = mmcv.imfrombytes(image)
            img = img[:, :, ::-1] # bgr --> rgb 
            img = img.transpose((2, 0, 1))  # h, w, c --> c, h, w
            images.append(img)
        
        return torch.from_numpy(np.array(images))

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
            'InternalInferenceTriangleTimeForBatch', 
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
        for ind, (triangles, translators) in enumerate(fetch_triangles_translators_batches(
                yolo_output=triangles_bboxes, 
                imgs=imgs, 
                n_batch=VOD_TRIANGLE_BATCHES,
                device=self.device
            )):
            
            vertices = vod_triangle_inference(triangles, *self.vod_triangle_models, device=self.device)

            for translator, triangle_vertices in zip(translators, vertices):
                tr_vert = triangle_vertices.tolist()
                bbox = triangles_bboxes[translator.img_index][translator.triangle_index]
                vert_dicts = convert_coords_list2dicts(tr_vert, translator)
                bbox, confidence = convert_yolo_output2dict(bbox)
                # print(tr_vert)
                # print(vert_dicts)
                # print(translator)
                img_dict = results[translator.img_index]
                img_dict["img_num"] = translator.img_index # has to be int because of sorting]

                img_dict["triangle_bbox"] = bbox
                if not "triangles" in img_dict:
                    img_dict["triangles"] = []
                img_dict["triangles"].append(
                    {
                        "triangle_num" : translator.triangle_index,
                        "triangle_vertices" : vert_dicts,
                        "yolo_confidence" : confidence
                    }
                )
        results = list((results.items()))
        results.sort(key=lambda a: a[0])
        print(f"TrianglesBatchSize.Batches:{len(triangles_bboxes)}")        
        idx = str(uuid.uuid4())
        self.context.metrics.add_time(
            'TrianglesInternalInferenceTimeForBatch', 
            (time.time() - tic) * 1000, 
            idx, 'ms'
        )
        

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
        metrics.add_metric('BatchSize', len(data), idx, 'Batches')


        tic = time.time()
        data_preprocess = self.preprocess(data)
        metrics.add_time('PreprocessingTimeForBatch', (time.time() - tic) * 1000, idx,  'ms')
        tic = time.time()
        if CUDA_ENABLED:
            with ILock("torchserve-mutex"):
                data_preprocess= data_preprocess.to(self.device)
                
                if not self._is_explain():
                    yolo_output = self.inference_triangles(data_preprocess)
                    vod_vertices = self.inference_vertices(data_preprocess, triangles_bboxes=yolo_output)
                else:
                    vod_vertices = self.explain_handle(data_preprocess, data)
                torch.cuda.empty_cache()
        else:
            data_preprocess= data_preprocess.to(self.device)
            if not self._is_explain():
                yolo_output = self.inference_triangles(data_preprocess)
                vod_vertices = self.inference_vertices(data_preprocess, triangles_bboxes=yolo_output)
            else:
                vod_vertices = self.explain_handle(data_preprocess, data)
            
        metrics.add_time('InferenceTimeForBatch', (time.time() - tic) * 1000, idx, 'ms')

        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), idx, 'ms')
        
        return vod_vertices

if __name__ == '__main__':
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
    
    img = open("./predictors/triangle/img.png","rb").read()

    tic = time.time()
    handler = TraingleHandler().initialize(Temp())
    results = handler.handle([{"data" : img} for _ in range(50)], Temp())
     
    #exit(0)
    print("Time for init", time.time() - tic)
    for _ in range(10):
        tic = time.time()
        results = handler.handle([{"data" : img} for _ in range(50)], Temp())
        toc = time.time()
        print("Handling Time", toc  - tic)
    # for res in results:
    #     import matplotlib.pyplot as plt
    #     import mmcv
    #     img = open("./predictors/triangle/img.png","rb").read()

    #     img = mmcv.imfrombytes(img)
    #     img = cv2.resize(img, (640,640))


    #     plt.imshow(img)
        
    #     xs, ys = [], []
        
    #     for vert in res["triangle_vertices"]:
    #         x_ = vert["x"]
    #         y_ = vert["y"]
    #         xs.append(x_)
    #         ys.append(y_)
    #     print(res)
    #     plt.scatter(xs,ys)
    # plt.savefig("./deleteme/giff.png")

    
