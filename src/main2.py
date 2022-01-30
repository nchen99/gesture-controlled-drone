# from tkinter import N
from unittest import runner
from utils.params import params
from model_processors.FaceDetectionProcessor import ModelProcessor as FaceDetectionProcessor
from model_processors.HandGestureProcessor import ModelProcessor as HandGestureProcessor
from atlas_utils.presenteragent import presenter_channel
from atlas_utils.acl_image import AclImage
from utils.RunLive import LiveRunner
import _thread

from atlas_utils.acl_resource import AclResource
import time
from enum import Enum
import cv2
from djitellopy import Tello
from utils.shared_variable import Shared
import threading
# from threading import Thread
from pid_controllers.run_track import init
from utils.runlive_2 import PresenterServer


shouldFollowMe: Shared

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
        # return State.INITIAL
    elif state == State.TAKEOFF_CONFIRM:
        return State.TAKEOFF if command == "2" else State.INITIAL
    elif state == State.TAKEOFF:
        return State.FLOAT
    elif state == State.FLOAT:
        if command == "1":
            return State.LAND_CONFIRM
            
        elif command == "3":
            return State.FOLLOW_ME_CONFIRM
        elif command == "4":
            return State.TAKE_A_PICTURE_CONFIRM
        else:
            return State.FLOAT
    elif state == State.LAND_CONFIRM:
        return State.LAND if command == "2" else State.LAND_CONFIRM
    elif state == State.LAND:
        return State.INITIAL
    elif state == State.FOLLOW_ME_CONFIRM:
        return State.FOLLOW_ME if command == "2" else State.FLOAT
    elif state == State.FOLLOW_ME:
        return State.STOP if command == "3" else State.FOLLOW_ME
    elif state == State.STOP:
        return State.FLOAT if command == "2" else State.FOLLOW_ME
    elif state == State.TAKE_A_PICTURE_CONFIRM:
        return State.TAKE_A_PICTURE if command == "2" else State.FLOAT
    elif state == State.TAKE_A_PICTURE:
        return State.FLOAT
    else:
        print(state)
        return state

def takeoff(tello, shouldFollowMe):
    tello.takeoff()

def land(tello, shouldFollowMe):
    tello.land()


following = False
def follow_me(_, shouldFollowMe):
    global following
    if not following:
        print("I am in follow me.")
        shouldFollowMe.set(True)
        # time.sleep(1)
        following = True

def floating(_, shouldFollowMe):
    global following
    if following:
        shouldFollowMe.set(False)
        following = False

def take_picture(tello, _):
    print("I am in taking pictures mode.")
    frame = tello.get_frame_read().frame
    cv2.imwrite("./picture.jpeg", frame)


state_to_func = {
    State.TAKEOFF: takeoff,
    State.LAND: land,
    State.FOLLOW_ME: follow_me,
    State.TAKE_A_PICTURE: take_picture,
    State.FLOAT: floating,
}

# tello = Tello()
# tello.connect()
# tello.takeoff()
# time.sleep(10)
# tello.land()

def runLive(p):
    p.main()

if __name__ == "__main__":
    for i in reversed(range(0, 15)):
        print(f"Starting drone in {i} seconds")
        time.sleep(1)
    
    tello = Tello()
    tello.connect()
    print(tello.get_battery(), "\n\n\n")
    tello.streamon()
    frame= tello.get_frame_read().frame
    print(frame)
    shouldFollowMe = Shared(False)

    _acl_resource = AclResource()
    _acl_resource.init()


    p = PresenterServer(tello)
    t1 = threading.Thread(target=init, args=(tello, shouldFollowMe, _acl_resource))
    t1.start()
    t2 = threading.Thread(target=runLive, args=(p,))
    t2.start()
    print("t2 start")
    state = State.INITIAL
    try:
        while True:
            func = state_to_func.get(state)
            if func is not None:
                print("Executing function related to state ", state)
                func(tello, shouldFollowMe)
            command = input("show the camera your body pose: ")
            state = get_next_state(state, command)
            print(state)
    except KeyboardInterrupt:
        tello.land()

