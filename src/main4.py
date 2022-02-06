from djitellopy import Tello
import openpose_tf
from atlas_utils.presenteragent import presenter_channel
from atlas_utils.acl_image import AclImage
import cv2
from utils.params import params
import time
from queue import Queue
from threading import Thread
# from atlas_utils.presenteragent import presenter_channel  

def show(chan, frame):
    _, jpeg_image = cv2.imencode('.jpg', frame)
    jpeg_image = AclImage(jpeg_image, frame.shape[0], frame.shape[1], jpeg_image.size)
    chan.send_detection_data(frame.shape[0], frame.shape[1], jpeg_image, [])


if __name__ == "__main__":
    uav_presenter_conf = params["presenter_server_conf"]
    chan = presenter_channel.open_channel(uav_presenter_conf)
    tello = Tello()
    tello.connect()
    print(tello.get_battery(), "\n\n\n")
    tello.streamon()
    frame= tello.get_frame_read().frame
    print(frame)


    openpose_tf.init(openpose_tf.MODEL_PATH)
    error= 0
    
    while True:
        tello.get_battery()
        img = tello.get_frame_read().frame
        if img is None:
            error += 1
            if error % 10 == 0:
                print("This many None images: ", error)
            continue
        results = openpose_tf.get_bounding_box(img)
        show(chan, img)
        if len(results) == 0:
            continue
        print(results[0]["area"])

    
