from logging import exception
from os import error
import requests
from threading import Thread
import base64
import time
import os

ip = "a3fa2d8a75e4a4d2dbdfe35f0875501c-17170527.us-east-2.elb.amazonaws.com:9001" # change this when having a new cluster
ip = "127.0.0.1:9001"
if os.environ.get("EKS", "False") == "True":
    ip = "torchserve-elb:9001"
    print("TARGET ELB IP IS", ip)
all_det = f"http://{ip}/predictions/triangle"
print(ip)
IMG_PATH = "./triangles_heavy_img.png"
IMG_COORDS =[[182, 38, 229, 102]]

def request_json():

    decompressed_image_bytes = open(IMG_PATH, "rb").read()
    #print(len(decompressed_image_bytes))
    response_raw = requests.put(all_det, data=decompressed_image_bytes, timeout=10)
    #print(response_raw.content)
    response_json = response_raw.json()
    
    return response_json, response_raw.elapsed.microseconds / 1e6

def equals( x, y):
        if isinstance(x, list):
            return all(equals(a,b) for a,b in zip(x,y))
        return abs(x - y) < 10
class Worker(Thread):

    def __init__(self, threadID, wait_time, image_path, test_time=60, debug=True, experiment="swarm"):
        Thread.__init__(self)
        self.threadID = threadID
        self.experiment = experiment
        self.result = None
        self.image_path = image_path
        self.cnt = 0
        self.wait_time = wait_time
        self.test_time = test_time
        self.logs = []
        self.debug = debug
        self.done = False
        self.run_time_start = None

    

    def run(self):
        error_log = []
        st = time.time()
        lst = st
        if self.debug:
            print(f"starting {self.threadID}")
       
        bst = time.time()
        time_elapsed = 0 
        try:
            self.run_time_start = time.time()
            blocks, time_elapsed = request_json()
            
            #assert equals(blocks, IMG_COORDS), " error from server side" + str(len(blocks)) + " " + str(blocks)
            #print(blocks.keys())
            assert "img_index" in blocks,f"assertion error blocks are {str(blocks)[:100]}"
            error_log.append({
                "thread_id": self.threadID,
                "experiment#": self.experiment,
                "status": True,
                "response_time": time.time() - bst,
                "msg": "success",
                "blocks": len(blocks)
            })
            #print("#########################################")
        except Exception as e:
            error_log.append({
                "thread_id": self.threadID,
                "experiment#": self.experiment,
                "status": False,
                "response_time": time_elapsed,
                "msg": "fail",
                "exception": str(e.args)
            })
            #print(str(e.args))
            #print(error_log[0]["exception"])
            self.done = True
        #time.sleep(self.wait_time)
        lst = bst
        if self.debug:
            print(error_log[-1])
        self.done = True
        self.logs = error_log
        self.log = error_log[0]
        return error_log


if __name__ == "__main__":
    tic = time.time()
    import json
    print(json.dumps(request_json()[0], indent=4, sort_keys=True))

    print("Time", time.time() - tic)
    print(equals( [[182, 38, 229, 105]], [[182, 38, 229, 102]]))
