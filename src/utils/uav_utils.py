from djitellopy import Tello
from curtsies import Input
from queue import Queue

def connect_uav():
    print("\n################################################################################")
    print("Connecting to Tello UAV...")
    try:
        uav = Tello()
        uav.connect()
        print("UAV connected successfully!")
        print(f"Current battery percentage: {uav.get_battery()}")
        return uav
    except Exception as e:
        print("Failed to connect to Tello UAV, please try to reconnect")
        raise 


def manual_control(uav, q):
    ## Add State Machine here
    # Gesture code to be used:
    # gesture_categories = [
    #     '0',
    #     '1',
    #     '2',
    #     '3',
    #     '4',
    #     '5',
    #     '6',
    #     '7',
    #     '8',
    #     '9',
    #     'left',
    #     'ok',
    #     'right',
    #     'rock',
    #     'finger heart',
    #     'praise',
    #     'prayer',
    #     'stop',
    #     'Give the middle finger',
    #     'bow',
    #     'No gesture'
    # ]
    while True:
        val = q.get()
        print(val)
        # if val == '3':
        #     uav.takeoff()
        # elif val == '2':
        #     uav.land() 