# from tkinter import N
from unittest import runner
from utils.params import params
from model_processors.BaseProcessor import BaseProcessor
from model_processors.FaceDetectionProcessor import ModelProcessor as FaceDetectionProcessor
from model_processors.HandGestureProcessor import ModelProcessor as HandGestureProcessor
from atlas_utils.presenteragent import presenter_channel
from atlas_utils.acl_image import AclImage
import _thread

from atlas_utils.acl_resource import AclResource
import time
from enum import Enum
import cv2
from djitellopy import Tello
from utils.shared_variable import Shared
import threading
# from threading import Thread
from utils.runlive import PresenterServer
import openpose_tf

from atlas_utils.acl_resource import AclResource
import os
from utils.process_2 import process_2
from multiprocessing import Process, Pipe

from utils.send_mail import send_mail

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

flight_id = "flight_id"
def takeoff(tello, _):
    global flight_id
    tello.takeoff()
    time.sleep(3)
    print("Moveing up")
    tello.move_up(30)
    flight_id = time.ctime().replace(" ", "_")
    os.mkdir("./outputs/{flight_id}")
    print("Flight ID:", flight_id)

def land(tello, _):
    tello.land()


following = False
def follow_me(tello, parent_conn):
    tello.takeoff()
    time.sleep(2)
    global following
    if not following:
        print("I am in follow me.")
        parent_conn.send("TRUE")
        following = True

def floating(_, parent_conn):
    global following
    if following:
        parent_conn.send("FALSE")
        following = False

def take_picture(tello, _):
    global flight_id
    print("I am in taking pictures mode.")
    frame = tello.get_frame_read().frame
    cv2.imwrite(f"./outputs/{flight_id}/{time.time_ns()}.png", frame)


# using email for now
# TODO: need debugging
# called when landing or on gestures?
# include args if on gestures
email = "shawnlu4@gmail.com"
def upload_images():
    global flight_id, email
    files = []
    for (dirpath, dirnames, filenames) in os.walk(f"./outputs/{flight_id}"):
        for photo in filenames:
            files.append(f"{dirpath}/{filenames}")
    send_mail(email, f"Your flight photos on {flight id}", "Taken by gesture-controlled vlogging assistant", files=files)


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
    
    # for i in reversed(range(0, 15)):
    #     print(f"Starting drone in {i} seconds")
    #     time.sleep(1)

    damn = time.time()
    time.sleep(5)
    damn = time.time() - damn
    
    tello = Tello()
    tello.connect()
    print(tello.get_battery(), "\n\n\n")
    tello.streamon()
    frame= tello.get_frame_read().frame
    print(frame)


    parent_conn, child_conn = Pipe()
    p = Process(target=process_2, args=(tello, child_conn,))
    p.start()

    # the rest:
    p = PresenterServer(tello)

    t2 = threading.Thread(target=runLive, args=(p,))
    t2.start()

    print("t2 start")
    state = State.FOLLOW_ME
    is_confirm = False
    confirm_timeout = None

    # print("wait pid to do its stuff, 10 sec")
    # time.sleep(10)
    openpose_tf.init(openpose_tf.MODEL_PATH)
    # time.sleep(5)

    try:
        while True:
            func = state_to_func.get(state)
            if func is not None:
                print("Executing function related to state ", state)
                func(tello, parent_conn)
            frame= tello.get_frame_read().frame
            command = openpose_tf.get_pose(frame)
            if len(command) > 0:
                command = str(command[0].value)
            else:
                command = "0"

            state = get_next_state(state, command)
            print(state, command)

            if state in [State.TAKEOFF_CONFIRM, State.FOLLOW_ME_CONFIRM, State.LAND_CONFIRM, State.TAKE_A_PICTURE_CONFIRM]:
                confirm_timeout = time.time() + damn
                # command = openpose_tf.get_pose(frame)
                while time.time() < confirm_timeout:
                    frame = tello.get_frame_read().frame
                    command = openpose_tf.get_pose(frame)
                    if len(command) > 0:
                        command = str(command[0].value)
                    else:
                        command = "0"
                    print(state, command)
                    if command == "2":
                        state = get_next_state(state, "2")
                        break
                    
                    
                    
            
                


            # if not is_confirm and next_state in [State.TAKEOFF_CONFIRM, State.FOLLOW_ME_CONFIRM, State.LAND_CONFIRM, State.TAKE_A_PICTURE_CONFIRM]:
            #     is_confirm = True
            #     confirm_timeout = time.time() + damn
            #     continue

            # if is_confirm and time.time() < confirm_timeout and command != "2":
            #     pass
            # else:
            #     state = next_state
            #     is_confirm = False
            # print(state)
            
    except KeyboardInterrupt:
        tello.land()
   
    



# if __name__ == "__main__":

#     # for i in reversed(range(0, 15)):
#     #     print(f"Starting drone in {i} seconds")
#     #     time.sleep(1)

#     damn = time.time()
#     time.sleep(3)
#     damn = time.time() - damn
    
#     tello = Tello()
#     tello.connect()
#     print(tello.get_battery(), "\n\n\n")
#     tello.streamon()
#     frame= tello.get_frame_read().frame
#     print(frame)


#     r, w = os.pipe()


#     processid = os.fork()
#     if processid:
#         # Parent ID:
#         os.close(r)
#         wr = os.fdopen(w, 'w')
#         # the rest:
#         p = PresenterServer(tello)

#         t2 = threading.Thread(target=runLive, args=(p,))
#         t2.start()

#         print("t2 start")
#         state = State.FOLLOW_ME
#         is_confirm = False
#         confirm_timeout = None

#         # print("wait pid to do its stuff, 10 sec")
#         # time.sleep(10)
#         openpose_tf.init(openpose_tf.MODEL_PATH)
#         # time.sleep(5)

#         try:
#             while True:
#                 func = state_to_func.get(state)
#                 if func is not None:
#                     print("Executing function related to state ", state)
#                     func(tello, wr, w)
#                 frame= tello.get_frame_read().frame
#                 command = openpose_tf.get_pose(frame)
#                 if len(command) > 0:
#                     command = str(command[0].value)
#                 else:
#                     command = "0"

#                 state = get_next_state(state, command)
#                 # print(state, command)

#                 if state in [State.TAKEOFF_CONFIRM, State.FOLLOW_ME_CONFIRM, State.LAND_CONFIRM, State.TAKE_A_PICTURE_CONFIRM]:
#                     confirm_timeout = time.time() + damn
#                     # command = openpose_tf.get_pose(frame)
#                     while time.time() < confirm_timeout:
#                         frame = tello.get_frame_read().frame
#                         command = openpose_tf.get_pose(frame)
#                         if len(command) > 0:
#                             command = str(command[0].value)
#                         else:
#                             command = "0"
#                         print(state, command)
#                         if command == "2":
#                             state = get_next_state(state, "2")
#                             break
                        
                        
                        
                
                    


#                 # if not is_confirm and next_state in [State.TAKEOFF_CONFIRM, State.FOLLOW_ME_CONFIRM, State.LAND_CONFIRM, State.TAKE_A_PICTURE_CONFIRM]:
#                 #     is_confirm = True
#                 #     confirm_timeout = time.time() + damn
#                 #     continue

#                 # if is_confirm and time.time() < confirm_timeout and command != "2":
#                 #     pass
#                 # else:
#                 #     state = next_state
#                 #     is_confirm = False
#                 # print(state)
                
#         except KeyboardInterrupt:
#             tello.land()
#     else:
#         os.close(w)
#         r = os.fdopen(r)
#         process_2(tello, r)

    

