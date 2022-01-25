from utils.params import params
from model_processors.FaceDetectionProcessor import ModelProcessor as FaceDetectionProcessor
from model_processors.HandGestureProcessor import ModelProcessor as HandGestureProcessor

from atlas_utils.acl_resource import AclResource
import time
from enum import Enum
import cv2
from djitellopy import Tello
from datetime import datetime

class State(Enum):
    INITIAL = 1
    TAKEOFF_CONFIRM = 2
    TAKEOFF = 3
    FLOAT = 4
    LAND_CONFIRM = 5
    LAND = 6
    FOLLOW_ME_CONFIRM = 7
    FOLLOW_ME = 8
    STOP = 9
    TAKE_A_PICTURE_CONFIRM = 10
    TAKE_A_PICTURE = 11

def get_next_state(state, command):
    if state == State.INITIAL: 
        return State.TAKEOFF_CONFIRM if command == "1" else State.INITIAL
    elif state == State.TAKEOFF_CONFIRM:
        return State.TAKEOFF if command == "2" else State.INITIAL
    elif state == State.TAKEOFF:
        return State.FLOAT
    elif state == State.FLOAT:
        return State.LAND_CONFIRM if command == "1" else State.FLOAT
    elif state == State.LAND_CONFIRM:
        return State.LAND if command == "2" else State.FLOAT
    elif state == State.LAND:
        return State.INITIAL
    else:
        print(state)
        return state

def takeoff(tello, face):
    tello.takeoff()

def land(tello, face):
    tello.land()

def follow_me(tello, face):
    print("I am in follow me.")
    time.sleep(1)

def take_picture(tello, face):
    frame = tello.get_frame_read().frame
    cv2.imwrite("./picture.jpeg", frame)


state_to_func = {
    State.TAKEOFF: takeoff,
    State.LAND: land,
}

if __name__ == "__main__":
    tello = Tello()
    tello.connect()
    print(tello.get_battery(), "\n\n\n")
    tello.streamon()

    _acl_resource = AclResource()
    _acl_resource.init()

    face = FaceDetectionProcessor(params["task"]["object_detection"]["face_detection"], _acl_resource)
    hand = HandGestureProcessor(params["task"]["classification"]["gesture_yuv"], _acl_resource)

    state = State.INITIAL
    while True:
        func = state_to_func.get(state)
        if func is not None:
            func(tello, face)
        frame = tello.get_frame_read().frame
        command = hand.predict(frame)
        state = get_next_state(state, command)
        # if state == State.TAKEOFF_CONFIRM:
            

        # print(command, state)
        time.sleep(0.5)


    face.release_acl()