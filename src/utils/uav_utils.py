from djitellopy import Tello
from curtsies import Input
from queue import Queue
import fsm
import drone_actions

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


process_state = {
    fsm.STATE_INITIAL: drone_actions.initial
}


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

    state = fsm.state_init()

    while True:
        gesture = q.get()
        print(gesture)

        state = fsm.next_state(state, gesture)

        process_state[state](uav)
        # if val == '3':
        #     uav.takeoff()
        # elif val == '2':
        #     uav.land() 