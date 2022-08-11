import sys

sys.path.append("../torchserve")
from  coordinator import MMdetHandler
import unittest
import time
import numpy as np

class Test_Coordinator(unittest.TestCase):
    def __init__(self, methodName="coordinator"):

        class Metrics:
            def add_time(self, *args, **kwargs):
                self.log(*args, **kwargs)
            def add_metric(self, *args, **kwargs):
                self.log(*args, **kwargs)
            def log(self, *args, **kwargs):
                print("Logger.args= ", *args)
                print("Logger.kwargs= ", *args)    

        class Context:
            system_properties = {
                "gpu_id" : "0"
            }

            metrics = Metrics()
            def get_request_header(self, *args, **kwargs):
                return True


        super().__init__(methodName)
        self.model_handler = MMdetHandler()
        self.model_handler.initialize(context=Context())
        self.handle = self.model_handler.handle
        self.context = Context()

    def test_batch(self, batch_sz=128, img_path = "../tests/influencer.png"):
        img_binary = open(img_path, "rb").read()
        json_req = {"body" : img_binary}
        batch = [json_req for _ in range(batch_sz)]
        print("Initializing")
        def handle_once():
            tic = time.time()
            res = self.handle(batch, self.context)
            toc = time.time()
            print("HandleTime", toc - tic)
            return res
        tms = []
        for _ in range(10):
            tic = time.time()
            resp = handle_once()
            toc = time.time()
            tms.append(toc - tic)
           
        tms = np.array(tms)
        print(tms.mean(), tms.max(), tms.min(), tms.std())
if __name__ == "__main__":
    unittest.main()