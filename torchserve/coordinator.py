from unittest import result
from PIL import Image
import uuid
import time

import numpy as np
import base64
import mmcv
import cv2
import torch

from methods.constants import TRIANGLE_MODEL_PATH, MAP_LOCATION
from predictors.triangle.infer_standalone import infer, load_model
 
from ts.torch_handler.base_handler import BaseHandler


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
        
        self.map_location = MAP_LOCATION
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        # assert self.device == "cuda", "GPU ISNT RECOGNIZED"
        print("DEVICE", self.device)
        self.triangle_model = load_model(TRIANGLE_MODEL_PATH, map_location=self.device)
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
            self.context.metrics.add_metric('PackageSize', len(image) // 8000, 'KB', str(uuid.uuid4()))
            if isinstance(image, str):
                image = base64.b64decode(image)
            img = mmcv.imfrombytes(image)
            img = img[:, :, ::-1] # bgr --> rgb 
            img = cv2.resize(img, (640, 640))
            img = img.transpose((2, 0, 1))  # h, w, c --> c, h, w
            
            images.append(img)
        
        return images

    def inference(self, data, *args, **kwargs):
        """
        Args:
            data: list of cv2 images
        Returns:
            
        """
        tic = time.time()
        results = []
        for img in data:
            results.append(infer(model=self.triangle_model, img=img))
        

        
        
        print(f"BatchSize.Batches:{len(data)}")        
        idx = str(uuid.uuid4())
        self.context.metrics.add_time(
            'InternalInferenceTimeForBatch', 
            (time.time() - tic) * 1000, 
            idx, 'ms'
        )

        return results

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
        if not self._is_explain():
            output = self.inference(data_preprocess)
        else:
            output = self.explain_handle(data_preprocess, data)
        metrics.add_time('InferenceTimeForBatch', (time.time() - tic) * 1000, idx, 'ms')

        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), idx, 'ms')
        
        return output

if __name__ == '__main__':
    class Metrics:
            def add_time(self, *args, **kwargs):
                pass
    class Temp:
        system_properties = {"gpu_id": "0"}
        
        metrics = Metrics()
        # def __str__(self):
        #     return ("here")
    print("tmp", Temp())
    img = open("./predictors/triangle/img.png","rb").read()
    imgs =TraingleHandler().initialize(Temp()).preprocess([{"data" : img}])

    print(TraingleHandler().initialize(Temp()).inference(imgs))
    time.sleep(10)
    
