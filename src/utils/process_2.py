from pid_controllers.run_track import init
from utils.shared_variable import Shared
import os


def process_2(tello, r):
    shouldFollowMe = Shared(False)

    init(tello, shouldFollowMe)


    while r.read():
        shouldFollowMe.set(not shouldFollowMe.get())


