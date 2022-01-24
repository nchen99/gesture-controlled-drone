import sys
import os
import time
from threading import Thread
from utils.uav_utils import connect_uav 
from utils.RunModels import RunModel
from pid_controllers.run_track import init
from utils.shared_variable import Shared


if __name__ == '__main__':
    # Connect to Tello drone
    uav = connect_uav()

    shouldFollowMe = Shared(False)

    t1 = Thread(target=RunModel, args=(uav, 0, shouldFollowMe))
    t1.start()


    t2 = Thread(target=init, args=(uav, shouldFollowMe))
    t2.start()

    t1.join()
    t2.join()