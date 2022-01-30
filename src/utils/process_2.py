from pid_controllers.run_track import init
from utils.shared_variable import Shared
import os
import threading
import time
from multiprocessing import Process, Pipe


def process_2(tello, child_conn):
    shouldFollowMe = Shared(False)
    t = threading.Thread(target=changed, args=(shouldFollowMe, child_conn))
    t.start()
    init(tello, shouldFollowMe)



def changed(shouldFollowMe, child_conn):
    while True:
        if child_conn.recv() == "TRUE":
            print("------------------------------------FOLLOW!!!!!!!!!!!!!!------------------------------------")
            shouldFollowMe.set(True)
        else:
            print("------------------------------------STOP!!!!!!!!!!!!!!------------------------------------")
            shouldFollowMe.set(False)