import sys
import os
import time
from utils.uav_utils import connect_uav 
from utils.RunLive import LiveRunner


if __name__ == '__main__':
    # Connect to Tello drone
    uav = connect_uav()
    SRC_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PRESENTER_SERVER_CONF = os.path.join(SRC_PATH, "uav_presenter_server.conf")
    print("HHHHHHH",PRESENTER_SERVER_CONF)

    _ = input("Press any key to start!")

    runner = LiveRunner(uav)
    # needed to fully connect to presenter server?
    time.sleep(10)
    runner.display_result()