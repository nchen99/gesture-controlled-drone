import sys
import os
import time
from threading import Thread
from utils.uav_utils import connect_uav 
from utils.RunModels import RunModel
from pid_controllers.run_track import init
from utils.shared_variable import Shared
from atlas_utils.acl_resource import AclResource


if __name__ == '__main__':
    try:
        # Connect to Tello drone
        uav = connect_uav()

        _acl_resource = AclResource()
        _acl_resource.init()

        shouldFollowMe = Shared(False)

        t1 = Thread(target=RunModel, args=(uav, shouldFollowMe, _acl_resource))
        t1.start()


        t2 = Thread(target=init, args=(uav, shouldFollowMe, _acl_resource))
        t2.start()

        t1.join()
    except:
        uav.land()
        pass
    # t2.join()