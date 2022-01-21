import sys
import os
import time
from utils.uav_utils import connect_uav 
from utils.RunLive import LiveRunner


if __name__ == '__main__':
    # Connect to Tello drone
    uav = connect_uav()

    runner = LiveRunner(uav)
    # needed to fully connect to presenter server?
    time.sleep(10)
    runner.display_result()