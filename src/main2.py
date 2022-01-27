# from tkinter import N
from utils.params import params
from model_processors.FaceDetectionProcessor import ModelProcessor as FaceDetectionProcessor
from model_processors.HandGestureProcessor import ModelProcessor as HandGestureProcessor
from utils.RunLive import LiveRunner

from atlas_utils.acl_resource import AclResource
import time
from enum import Enum
import cv2
from djitellopy import Tello
from utils.shared_variable import Shared
from threading import Thread
from pid_controllers.run_track import init

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
        # return State.TAKEOFF_CONFIRM if command == "1" else State.INITIAL
        return State.INITIAL
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
        time.sleep(1)
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

def run_live(tello, shouldFollowMe, _acl_resource):
    runner = LiveRunner(tello, shouldFollowMe, _acl_resource)
    # needed to fully connect to presenter server?
    time.sleep(10)
    runner.display_result()



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

if __name__ == "__main__":
    tello = Tello()
    tello.connect()
    print(tello.get_battery(), "\n\n\n")
    tello.streamon()


    # while True:
    # frame = tello.get_frame_read().frame
        

    # while True:
        # try:
        #     frame = tello.get_frame_read().frame
        #     cv2.imshow("drone", frame)
        #     cv2.waitKey(1)
        # except:
        #     pass

    # _acl_resource = AclResource()
    # _acl_resource.init()

    shouldFollowMe = Shared(False)

    run_live(tello, shouldFollowMe, "random arg") # Third arg should be _acl_resource but doesn't matter?

    # runner = LiveRunner(tello, shouldFollowMe, _acl_resource)
    # # needed to fully connect to presenter server?
    # time.sleep(10)
    # runner.display_result()

    t1 = Thread(target=init, args=(tello, shouldFollowMe))
    # t2 = Thread(target=run_live, args=(tello, shouldFollowMe, "a"))
    
    # t2.start()
    # time.sleep(10)
    t1.start()
    

    # face = FaceDetectionProcessor(params["task"]["object_detection"]["face_detection"], _acl_resource)
    # hand = HandGestureProcessor(params["task"]["classification"]["gesture_yuv"], _acl_resource)

    state = State.INITIAL

    # cur = time.process_time()
    # time.sleep(3)
    # threadshold = time.process_time() - cur
    # command = 
    while True:
        try: 


            func = state_to_func.get(state)
            if func is not None:
                print("Executing function related to state ", state)
                func(tello, shouldFollowMe)

            if state == State.LAND_CONFIRM or state == State.TAKEOFF_CONFIRM:
                start = time.time()
                print(f"Entering state {state}, waiting to confirm...")
                while time.time() - start < 10:
                    command = input("show the camera your body pose: ")
                    if command == "2":
                        print("Confirmed")
                        state = get_next_state(state, command)
                        print(state)
                        break
            
            
            else:
                command = input("show the camera your body pose: ")
                state = get_next_state(state, command)
                print(state)
            # if state == State.TAKEOFF_CONFIRM:
            

        except KeyboardInterrupt as e:
            # face.released_acl()
            tello.land()
            tello.streamoff()
            t1.join()
            # t2.join()
