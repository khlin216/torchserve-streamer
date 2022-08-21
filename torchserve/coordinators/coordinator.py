from unittest import result
from PIL import Image
import uuid
import time

import numpy as np
import mmcv

from shutil import move, rmtree
from methods.face_det_init_cnn import create_mtcnn, create_resnet
from ts.torch_handler.base_handler import BaseHandler
from ts.metrics.dimension import Dimension
import base64

from methods.constants import *


print("STUFF IN THE DIRECTORY")


class MMdetHandler(BaseHandler):
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
        self.device = MAP_LOCATION
        # assert self.device == "cuda", "GPU ISNT RECOGNIZED"
        self.mtcnn = create_mtcnn(MAP_LOCATION)
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
            image = mmcv.imfrombytes(image)
            img_np = image[:, :, ::-1]
            images.append(img_np)
        
        return images

    def inference(self, data, *args, **kwargs):
        """
        Args:
            data: list of cv2 images
        Returns:
            
        """
        tic = time.time()

        boxes_list, confidence_list = self.mtcnn.detect(data)
        

        results = []
        for boxes in boxes_list:
            result = []
            results.append(result)
            if boxes is None:
                continue
            for box in boxes:
                result.append([int(round(i)) for i in box])
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
    # class Temp:
    #     system_properties = {"gpu_id": "0"}
    # MMdetHandler().initialize(Temp())
    pass