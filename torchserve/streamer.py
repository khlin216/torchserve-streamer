import streamlink
import time
from facenet_pytorch import MTCNN
device = 'cuda'


#!/usr/bin/env python3
import logging
import sys
import streamlink
import os.path
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


def detect_faces(frame, mtcnn):
    boxes, _ = mtcnn.detect(frame)
    for box in boxes:
        box = [int(i) for i in box]
        x1, y1, x2, y2 = box
        #assert 0, box
        print(box, frame.shape)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 3)
    return frame

def main(url, quality='best', fps=60.0):

    
    mtcnn = MTCNN(
        image_size=1920, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    
    
    stream_url = stream_to_url(url)
    log.info("Loading stream {0}".format(stream_url))
    cap = cv2.VideoCapture(stream_url)

    frame_time = int((1.0 / fps) * 1000.0)
    width  = int(cap.get(3))
    height = int(cap.get(4))
     
    

    frames_queue = []
    while True:
        try:
            ret, frame = cap.read()
            print(len(frames_queue))
            if ret:
                frames_queue.append(frame)
                frame_f = detect_faces( frame, mtcnn)
                cv2.imshow('frame', frame_f)
                
                if cv2.waitKey(frame_time) & 0xFF == ord('q'):
                    break
            else:
                break
        except KeyboardInterrupt:
            break

   
    print("-=========================-")
    for frame in frames_queue:
        try:
            #ret, frame = cap.read()
            
            
            #frames_queue.append(frame)
            frame_f = frame
            cv2.imshow('frame', frame_f)
            
            if cv2.waitKey(frame_time) & 0xFF == ord('q'):
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