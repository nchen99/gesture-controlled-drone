import _thread
from multiprocessing import shared_memory
import time
from utils.RunLive import LiveRunner
from utils.shared_variable import Shared

def RunModel(self, uav, model_number, shouldFollowMe):

        runner = LiveRunner(uav, model_number, shouldFollowMe)
        # needed to fully connect to presenter server?
        time.sleep(10)
        runner.display_result()

        



