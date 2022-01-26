import time
from utils.RunLive import LiveRunner

def RunModel(uav, shouldFollowMe, _acl_resource):

        runner = LiveRunner(uav, shouldFollowMe, _acl_resource)
        # needed to fully connect to presenter server?
        time.sleep(10)
        runner.display_result()

        



