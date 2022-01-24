import time
from utils.RunLive import LiveRunner

def RunModel(uav, shouldFollowMe):

        runner = LiveRunner(uav, shouldFollowMe)
        # needed to fully connect to presenter server?
        time.sleep(10)
        runner.display_result()

        



