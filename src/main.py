import sys
import os
import time
import _thread
from utils.uav_utils import connect_uav 
from utils.RunModels import RunModel
from pid_controllers import run_track
from utils.shared_variable import Shared


if __name__ == '__main__':
    # Connect to Tello drone
    uav = connect_uav()

    shouldFollowMe = Shared(False)

    _thread.start_new_thread(RunModel, (uav, 0, shouldFollowMe)) #span hand gesture recognition
    _thread.start_new_thread(run_track.init, (uav, shouldFollowMe)) #span follow me 