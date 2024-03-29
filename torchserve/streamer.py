#!/usr/bin/env python3

import streamlink
import time
from facenet_pytorch import MTCNN
import io
import requests
import concurrent.futures
import requests
import time
import torch
import sys
import logging
import sys
import streamlink
import os.path
import json
import matplotlib.pyplot as plt
import numpy as np
from methods.constants import IMG_HEIGHT, IMG_WIDTH
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    import cv2
except ImportError:
    sys.stderr.write("This example requires opencv-python is installed")
    raise

log = logging.getLogger(__name__)
GREEN = (0, 255, 0)
EKS_IP = "http://ac1079231337f47aabdc6aa6e7a2be07-233993352.us-east-2.elb.amazonaws.com" 
# EKS_IP =  "http://127.0.0.1" # 

def stream_to_url(url, quality='best'):
    if "twitch" in url:
        streams = streamlink.streams(url)
        if streams:
            return streams[quality].to_url()
        else:
            raise ValueError("No streams were available")
    else:
        return url


def add_rect2frame(frame, boxes):
    for box in boxes:
        box = [int(i) for i in box]
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)


def augment(frame, boxes, asset):
    for box in boxes:
        x0, y0, x1, y1 = [int(i) for i in box]
        if x1 - x0 <= 0:
            continue
        if y1 - y0 <= 0:
            continue
        asset_patch = cv2.resize(asset, (x1 - x0, y1 - y0))
        a = (asset_patch[:, :, 3] / 255.)[:, :, None]
        frame[y0: y1, x0: x1] = (1-a) * frame[y0: y1, x0: x1] + a * asset_patch[:, :, :3]


def detect_faces(frame, mtcnn):
    boxes, _ = mtcnn.detect(frame)
    add_rect2frame(frame, boxes)
    return frame


def numpy_to_binary(arr):
    is_success, buffer = cv2.imencode(".png", arr)
    io_buf = io.BytesIO(buffer)
    return io_buf.read()


def detect_faces_online(frame, add2frame=False, timeout=5):
    X_sz, Y_sz = frame.shape[:2]
    W_new, H_new = IMG_WIDTH, IMG_HEIGHT
    resized = cv2.resize(frame, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
    r = requests.put(
            f"{EKS_IP}:9001/predictions/all_det",
            numpy_to_binary(resized), 
            timeout=timeout
        ).content

    scale_X = Y_sz / W_new
    scale_Y = X_sz / H_new

    boxes = json.loads(r.decode())
    #assert 0, boxes
    if isinstance(boxes, dict):
        print(boxes)
        exit(1)
        boxes = []
    boxes = [[x1 * scale_X, y1 * scale_Y , x2* scale_X, y2 * scale_Y] for x1, y1, x2, y2 in boxes]
    if add2frame:
        add_rect2frame(frame, boxes)
    return boxes


def write_on_line(text):
    sys.stdout.write(f'\r{text}')
    sys.stdout.flush()


def main(url, fpath_asset=None, x0=None, y0=None, x1=None, y1=None, quality='best', fps=300.0):
    stream_url = stream_to_url(url)
    log.info("Loading stream {0}".format(stream_url))
    cap = cv2.VideoCapture(stream_url)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("shape=",  (h, w))
    if x0 is None:
        x0 = 0.0
    if x1 is None:
        x1 = 1.0
    if y0 is None:
        y0 = 0.0
    if y1 is None:
        y1 = 1.0
    x0, x1 = int(round(x0 * w)), int(round(x1 * w))
    y0, y1 = int(round(y0 * h)), int(round(y1 * h))
    print(y0, y1, x0, x1)
    if fpath_asset is None:
        fpath_asset = "img.png"
        img_data = requests.get("http://assets.stickpng.com/images/58e8ff52eb97430e819064cf.png").content
        with open(fpath_asset, 'wb') as handler:
            handler.write(img_data)

    asset = cv2.imread(fpath_asset, cv2.IMREAD_UNCHANGED)

    frame_time = int((1.0 / fps) * 1000.0)
    CONNECTIONS = 200
    multithreader = concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS)
    tic = None 
    cnt = 0
    futures = []
    frames_queue = []
    beginning = True
    boxes = InertialBoxes(b=0.90)
    lst_image = np.ones((IMG_WIDTH, IMG_HEIGHT, 3))
    while True:
        try:
            tic_ = time.time()
            ret, frame = cap.read()
            print(f"Getting frame {time.time() - tic_}")
            if ret is None:
                break
            frame = frame[y0:y1, x0:x1]
            assert len(frame.ravel()) > 0
            frames_queue.append(frame)
            # frame = detect_faces_online(frame)
            futures.append(multithreader.submit(detect_faces_online, frame))
            print("A", time.time() -tic_)
            try:
                print("Futures# = ", len(futures))
                if len(futures) < 100 and beginning:
                    print('=================')
                    continue
                beginning = False
                if tic is None:
                    tic = time.time()
                future = next(concurrent.futures.as_completed(futures[0:1]))
                print("B", time.time() -tic_)
                boxes_new = future.result()
                print("B.1")
                boxes.tick()
                for box in boxes_new:
                    boxes.handle_box(box)
                futures.pop(0)
                frame = frames_queue.pop(0)
                print("C", time.time() -tic_)
                # add_rect2frame(frame, boxes)
                augment(frame, [box["coords"] for box in boxes.info], asset)
                print("D", time.time() -tic_)
                cv2.imshow('frame', cv2.resize(frame, (1920//2, 1080//2)))
                lst_image = frame
                cnt += 1
                print("E", time.time() -tic_)

            except Exception as e:
                #print(e.args)
                print("Err#1")
                cv2.imshow('frame', cv2.resize(lst_image, (1920//2, 1080//2)))
                cnt += 1
                #exit(1)
                #time.sleep(100)

            # time.sleep(100)
            toc = time.time()
            print(f"FPS = {cnt / (toc - tic)}")
            if cv2.waitKey(max(1, min(1, frame_time - 1000 * int(toc - tic_)))) & 0xFF == ord('q'):
                break
            print("F", time.time() -tic_)
        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()
    cap.release()


def stream_without_torch(url, quality='best', fps=300.0):
    stream_url = stream_to_url(url)
    log.info("Loading stream {0}".format(stream_url))
    cap = cv2.VideoCapture(stream_url)

    frame_time = int((1.0 / fps) * 1000.0)
    tic = time.time()
    cnt = 0
    tic = time.time()
    while True:
        try:
            ret, frame = cap.read()
            
            if ret:
                tic_ = time.time()
                cv2.imshow('frame', frame)
                    
                cnt += 1
                # time.sleep(100)
                toc = time.time()
                print(f"FPS = {cnt / (toc - tic)}")
                if cv2.waitKey(max(0, frame_time - 1000 * int(toc - tic_))) & 0xFF == ord('q'):
                    break
            else:
                break
        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()
    cap.release()


class InertialBoxes:
    def __init__(self, b=0.90, tol=10, min_freshness=0.10, max_freshness=1.0):
        self.b = b
        self.tol = tol
        self.info = []
        self.min_freshness = min_freshness
        self.max_freshness = max_freshness

    def add_new_box(self, coords):
        x0, y0, x1, y1 = coords
        self.info.append({
            "center": (0.5 * (x1+x0), 0.5 * (y1+y0)),
            "coords": coords,
            "freshness": 1,
        })

    def update_box(self, i, coords):
        box = self.info[i]
        b = self.b
        box["coords"] = tuple(b * np.array(box["coords"]) + (1 - b) * np.array(coords))
        x0, y0, x1, y1 = box["coords"]
        box["center"] = (0.5 * (x1+x0), 0.5 * (y1+y0))
        box["freshness"] = min(box["freshness"] + 1, self.max_freshness)

    def tick(self):
        for i, box in enumerate(self.info):
            box["freshness"] *= self.b
        self.info = [box for box in self.info if box["freshness"] > self.min_freshness]

    def is_close(self, coords0, coords1):
        return abs(np.array(coords0) - np.array(coords1)).mean() < self.tol

    def handle_box(self, coords):
        found_box = False
        for i, box in enumerate(self.info):
            if self.is_close(coords, box["coords"]):
                found_box = True
                self.update_box(i, coords)
                break
        if not found_box:
            self.add_new_box(coords)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Face detection on streams via Streamlink")
    parser.add_argument("url", help="Stream to play")
    parser.add_argument("--path_asset", default=None)
    parser.add_argument("--x0", default=None, type=float)
    parser.add_argument("--y0", default=None, type=float)
    parser.add_argument("--x1", default=None, type=float)
    parser.add_argument("--y1", default=None, type=float)

    opts = parser.parse_args()
    
    TWITCH_URL = opts.url if opts.url else "https://www.twitch.tv/valhalla_cup"
    main(TWITCH_URL, opts.path_asset, opts.x0, opts.y0, opts.x1, opts.y1)
    # stream_without_torch(TWITCH_URL)
