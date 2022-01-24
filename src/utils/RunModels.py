import time
from utils.RunLive import LiveRunner

def RunModel(uav, model_number, shouldFollowMe):

        runner = LiveRunner(uav, model_number, shouldFollowMe)
        # needed to fully connect to presenter server?
        time.sleep(10)
        runner.display_result()

        



