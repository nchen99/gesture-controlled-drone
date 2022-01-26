from djitellopy import Tello

'''
    This file contains the actions associated with each state
    Actions should be either short or run in a thread that can be terminated on demand
'''







# STATE_INITIAL = 0
# STATE_TAKEOFF_CONFIRMATION = 1
# STATE_TAKEOFF = 2
# STATE_FLOAT = 3
# STATE_LAND_CONFIRMATION = 4
# STATE_LAND = 5
# STATE_FOLLOW_ME_CONFIRMATION = 6
# STATE_FOLLOW_ME = 7
# STATE_STOP_CONFIRMATION = 8
# STATE_TAKE_PICTURE_CONFIRMATION = 9
# STATE_TAKE_PICTURE = 10

# for development purposes,
uav = Tello()

thread = ""

# remember to add uav as arguments after
def initial():
    return


def take_off():
    uav.takeoff()
    return


'''
Calibration
lighting
patterned ground

crop hand
LED => help locating


'''