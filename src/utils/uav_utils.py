from djitellopy import Tello
from curtsies import Input
from utils.shared_variable import Shared
import time

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


def manual_control(uav, shouldFollowMe, command):
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
    i = 0
    while True:
        time.sleep(0.1)
        val = command.get()
        i += 1
    
        if i % 10 == 1:
            print(val)
        
        if val == '3':
            try:
                uav.takeoff()
            except:
                pass
        elif val == '2':
            try:
                uav.land()
            except:
                pass 
        elif val == '1':
            # add code here to detect if the drone has taken off or not:
            shouldFollowMe.set(True)
        elif val == '0':
            shouldFollowMe.set(False)
            