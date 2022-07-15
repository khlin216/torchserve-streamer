from unittest import result
from PIL import Image


import numpy as np
import cv2
import mmcv

from shutil import move, rmtree
from methods.face_det_init_cnn import create_mtcnn, create_resnet
from ts.torch_handler.base_handler import BaseHandler
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
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        assert self.device == "cuda", "GPU ISNT RECOGNIZED"
        self.mtcnn = create_mtcnn()

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
            image = mmcv.imfrombytes(image)
            img_np = image[:, :, ::-1]
            images.append(img_np)
        return images

    def inference(self, data, *args, **kwargs):
        """
        Args:
            data: list of cv2 images
        Returns:
            textract-like response representing the table/cell/text structure in the image
        """
        results = []
        for frame in data:
            # REPLACE WITH PREDICT BATCH IF POSSIBLE #TODO
            
            boxes, _ = self.mtcnn.detect(frame)
            results.append([])
            if boxes is None:
                continue
            for box in boxes:
                box = [int(i) for i in box]
                x1, y1, x2, y2 = box
                results[-1].append((x1, y1, x2, y2))
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
        import time
        start_time = time.time()
        self.context = context
        metrics = self.context.metrics
        data_preprocess = self.preprocess(data)
        if not self._is_explain():
            output = self.inference(data_preprocess)
        else:
            output = self.explain_handle(data_preprocess, data)
        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), None, 'ms')
        return output


if __name__ == '__main__':
    # class Temp:
    #     system_properties = {"gpu_id": "0"}
    # MMdetHandler().initialize(Temp())
    pass
