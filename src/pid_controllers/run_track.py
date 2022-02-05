"""
Closed-Loop Detection + Tracking System
Control relies on feedback from in a closed-loop manner - enable drone to automatically adjust itself without user intervention to detect and track an
object of interest

+ Visual Servoing with Robotics with object detection

"""
import time
import sys
import os
import cv2
from importlib import import_module
from datetime import datetime
import argparse

sys.path.append("..")
sys.path.append("../lib")

from utils.params import params
from atlas_utils.presenteragent import presenter_channel
from atlas_utils.acl_image import AclImage


def init_presenter_server():
    SRC_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PRESENTER_SERVER_CONF = os.path.join(SRC_PATH, "uav_presenter_server.conf")
    chan = presenter_channel.open_channel(PRESENTER_SERVER_CONF)
    if chan is None:
        raise Exception("Open presenter channel failed")
    return chan

def gen_save_name(tracker):
    now = datetime.now()
    save_name = now.strftime("%m%d%Y_%H%M%S")
    return f"PID_{tracker}_{save_name}"

def _init_tracker(tracker_name, **kwargs):
    """Imports corresponding PIDTracker based on tracker_name
    :param:
        + tracker_name - name of PIDTracker - currently supports "PIDFaceTracker" and "PIDPersonTracker"
        Acceptable kwargs: pid, save_flight_hist, inference_filter, if_fps, if_window, model_processor, save_flight_hist
    Returns
        instantiated PIDTracker object
    """
    tracker_module = import_module(f"pid_controllers.{tracker_name}")
    tracker_class = getattr(tracker_module, tracker_name)
    return tracker_class(**kwargs)

def _init_filter(filter_name, **kwargs):
    """Imports corresponding Inference Filter based on filter_name
    :param:
        + filter_name - name of inference fitler - currently supports "DecisionFilter"
        Acceptable kwargs: fsp (int), window (int)
    Returns
        instantiated InferenceFilter object
    """
    inference_filter_module = import_module(f"inference_filters.{filter_name}")
    filter_class =  getattr(inference_filter_module, filter_name)
    return filter_class(**kwargs)

def initialize_tracker(args):
    inference_filter = _init_filter(filter_name=args.inference_filter, fps=args.if_fps, window=args.if_window)
    tracker = _init_tracker(tracker_name=args.tracker, pid=args.pid, inference_filter=inference_filter)
    tracker.init_uav()
    return tracker

def parser():
    parser = argparse.ArgumentParser(description="Tello UAV PID-Tracker Setting")
    parser.add_argument("--flight_name", help="Flight run name", default=None)
    parser.add_argument("--use_ps",  type=bool, help="Forward flight video to Presenter Server if True", default=False)
    parser.add_argument("--duration", "-d", type=int, help="Flight duration (in seconds)", default=120)
    parser.add_argument("--tracker", "-t", help="Tracker name (i.e.: PIDFaceTracker)", default="PIDFaceTracker")
    parser.add_argument("--pid", nargs="+", help="PID List", default=[0.1, 0.1, 0.1])
    parser.add_argument("--save_flight", type=bool, help="Save flight statistics to pkl if True", default=False)
    parser.add_argument("--inference_filter", type=str, help="Inference Filter name", default="DecisionFilter")
    parser.add_argument("--if_fps", type=int, help="Incoming FPS for inference filter", default=5)
    parser.add_argument("--if_window", type=int, help="Observation window for inference filter", default=3)

    args = parser.parse_args()
    return args

def send_to_presenter_server(chan, frame_org, result_img):
    _, jpeg_image = cv2.imencode('.jpg', result_img)
    jpeg_image = AclImage(jpeg_image, frame_org.shape[0], frame_org.shape[1], jpeg_image.size)
    chan.send_detection_data(frame_org.shape[0], frame_org.shape[1], jpeg_image, [])

if __name__ == "__main__":
    args = parser()

    if args.use_ps:
        chan = init_presenter_server()

    x_err, y_err = 0, 0
    tookoff, flight_end = False, False
    timeout = time.time() + args.duration

    tracker = initialize_tracker(args)

    while not flight_end:
        try:
            if not tookoff:
                tookoff = True
                tracker.uav.takeoff()
                tracker.uav.move_up(70)
            
            if time.time() > timeout:        
                tracker.uav.land()
                tracker.uav.streamoff()
                flight_end = True
            
            frame_org = tracker.fetch_frame()
            x_err, y_err, result_img = tracker.run_state_machine(frame_org, x_err, y_err)

            if args.use_ps:
                send_to_presenter_server(chan, frame_org, result_img)

        except (KeyboardInterrupt, Exception) as e:
            tracker.uav.land()
            tracker.uav.streamoff()
            print(e)
            break

    if args.save_flight:
        if not os.path.exists("flights_data"):
            os.mkdir("flights_data")
        save_file = gen_save_name(args.tracker) if args.flight_name is None else args.flight_name
        tracker.save_hist(save_file)

