#!/usr/bin/env python3

import streamlink
import time
from facenet_pytorch import MTCNN
import io
import requests
import concurrent.futures
import requests
import time
device = 'cuda'

import sys
import logging
import sys
import streamlink
import os.path
import json
import matplotlib.pyplot as plt
try:
    import cv2
except ImportError:
    sys.stderr.write("This example requires opencv-python is installed")
    raise

log = logging.getLogger(__name__)
GREEN = (0, 255, 0)


def stream_to_url(url, quality='best'):
    streams = streamlink.streams(url)
    if streams:
        return streams[quality].to_url()
    else:
        raise ValueError("No streams were available")


def add_rect2frame(frame, boxes):
     
    for box in boxes:
         
        box = [int(i) for i in box]
        x1, y1, x2, y2 = box
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 3)


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
    resized = cv2.resize(frame, (100, 64), interpolation= cv2.INTER_LINEAR)
    r = requests.put(
            "http://127.0.0.1:9001/predictions/all_det", 
            numpy_to_binary(resized), 
            timeout=timeout
        ).content
    scale_X = Y_sz / 100 
    scale_Y = X_sz / 64

    boxes = json.loads(r.decode())
    boxes = [[x1 * scale_X, y1 * scale_Y , x2* scale_X, y2 * scale_Y] for x1, y1, x2, y2 in boxes]
    if add2frame:
        add_rect2frame(frame, boxes)
    return boxes


def write_on_line(text):
    sys.stdout.write(f'\r{text}')
    sys.stdout.flush()


def main(url, quality='best', fps=60.0):
    stream_url = stream_to_url(url)
    log.info("Loading stream {0}".format(stream_url))
    cap = cv2.VideoCapture(stream_url)

    frame_time = int((1.0 / fps) * 1000.0)
    CONNECTIONS = 200
    multithreader = concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS)
    tic = None 
    cnt = 0
    futures = []
    frames_queue = []
    beginning = True
    while True:
        try:
            tic_ = time.time()
            ret, frame = cap.read()
            print(f"Getting frame {time.time() - tic_}")
             
            if ret:
                
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
                    boxes = future.result()
                    futures.pop(0)
                    frame = frames_queue.pop(0)
                    print("C", time.time() -tic_)
                    print(boxes)
                    
                    add_rect2frame(frame, boxes)
                    print("D", time.time() -tic_)
                    
                    cv2.imshow('frame', frame)
                    cnt += 1
                    print("E", time.time() -tic_)
                    
                except Exception as e:
                    print(str(e.args))
                    exit(1)
                    #time.sleep(100)
                
                # time.sleep(100)
                toc = time.time()
                print(f"FPS = {cnt / (toc - tic)}")
                if cv2.waitKey(min(1, frame_time - 1000 * int(toc - tic_))) & 0xFF == ord('q'):
                    break
                print("F", time.time() -tic_)
            else:
                break
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


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Face detection on streams via Streamlink")
    parser.add_argument("url", help="Stream to play")

    opts = parser.parse_args()
    
    TWITCH_URL = opts.url if opts.url else "https://www.twitch.tv/valhalla_cup"
    main(TWITCH_URL)
    # stream_without_torch(TWITCH_URL)
