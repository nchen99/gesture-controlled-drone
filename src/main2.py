from utils.params import params
from model_processors.FaceDetectionProcessor import ModelProcessor as FaceDetectionProcessor
from model_processors.HandGestureProcessor import ModelProcessor as HandGestureProcessor

from djitellopy import Tello
import cv2
from atlas_utils.acl_resource import AclResource
import time


if __name__ == "__main__":
    tello = Tello()
    tello.connect()
    print(tello.get_battery())
    tello.streamon()

    _acl_resource = AclResource()
    _acl_resource.init()

    face = FaceDetectionProcessor(params["task"]["object_detection"]["face_detection"], _acl_resource)
    hand = HandGestureProcessor(params["task"]["classification"]["gesture_yuv"], _acl_resource)

    while True:
        frame = tello.get_frame_read().frame
        frame, _, boxList = face.predict(frame)
        frame, fk = hand.predict(frame)
        cv2.imshow("drone", frame)
        cv2.waitKey(1)
        time.sleep(0.1)
    # print("hand: ", i, fk)
    # print("face: ", boxList)

    face.release_acl()
    # tello.streamoff()