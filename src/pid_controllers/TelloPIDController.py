import time
import sys
import os
import cv2
import numpy as np
from importlib import import_module
from abc import abstractmethod
from djitellopy import Tello
import pickle
import argparse

sys.path.append("..")
sys.path.append("../lib")

from utils.params import params
from atlas_utils.presenteragent import presenter_channel
from atlas_utils.acl_image import AclImage

class TelloPIDController:
    """
    :class: TelloPIDController - Base class 
        Uses object detection models as a tracker. Process inference results and transform them into Process Variables for closed-loop control.
        
        :input:
            + PID (Proportion-Integral-Derivative) List(Floats) i.e.: [0.1, 0.1, 0.1]
        
        Setpoints:
            + BBox Center (x, y) = (480, 360); Tello frame size = (960, 720)
            + BBox Area 
        :components:
            + bbox_compensator - internal method for calculating distance b/w detected bbox of ToI and central point
            + Set-point        - pre-set attribute based on Tello data stream dimensions
            + Actuator         - internal method to stabilize droen and track ToI based on bbox_compensator's output
                            I.e.: compensator says bbox is far to the right, actuator rotate camera to the right to adjust

            + Accepts a connected TelloUAV - takeoff and streamon;
                - once stream is stablized run OD, 
    """
    detectors = params["task"]["object_detection"]

    def __init__(self, pid=[0.1, 0.1, 0.1], save_flight_hist=False):
        self.pid = pid
        self.save_flight_hist = save_flight_hist
        self.setpoint_center = (480, 360)
        self.history = []
        self.search_mode = True
        self.track_mode = False

    @staticmethod
    def _load_mp(detector_name):
        """Internal method for children class to load specific ModelProcessor
        :param:
            + detector_name - Key name of detection model
        Returns
            A fully initialized ModelProcessor object
        """
        model_info = TelloPIDController.detectors[detector_name]
        processor = model_info["model_processor"]
        MP = import_module(f"model_processors.{processor}")
        MP = getattr(MP, "ModelProcessor")
        return MP(model_info)
    
    @staticmethod
    def _load_filter(Filter, **kwargs):
        """Internal method to initialize and load an Inference Filter
        :param:
            filter_name - a Filter Object (i.e. DecisionFilter)
        Returns
            an initialized Filter object
        """
        inference_filter = Filter(**kwargs)
        return inference_filter

    def init_uav(self):
        """Initiate closed-loop tracking sequence, drone takeoff and parallelize streaming and control
        Returns
            None
        """
        try:
            self.uav = Tello()
            self.uav.connect()
            print("UAV connected successfully!")
            print(f"Current battery percentage: {self.uav.get_battery()}")
            self.uav.streamoff()
            self.uav.streamon()   
        except Exception as e:
            raise Exception("Failed to connect to Tello UAV, please try to reconnect")

    def fetch_frame(self):
        frame = self.uav.get_frame_read().frame
        return frame

    def _get_feedback(self, frame):
        """Obtains feedback (inference result) from model. Preprocess and execute the model using ModelProcessor.  
        :param:
            + frame - input frame for inference
        Returns
            Model's inference output (i.e: a list containing inference information such as bbox, num_detections, etc.)
        """
        preprocessed = self.model_processor.preprocess(frame)
        infer_output = self.model_processor.model.execute([preprocessed, self.model_processor._image_info])
        return infer_output
    
    def _pid(self, error, prev_error):
        """PID Output signal equation"""
        return self.pid[0]*error + self.pid[1]*(error-prev_error) + self.pid[2]*(error-prev_error)

    @abstractmethod
    def _unpack_feedback(self, inference_info, frame):
        pass

    @abstractmethod
    def _track(self, inference_info):
        pass

    @abstractmethod
    def _search(self):
        pass

    @abstractmethod
    def _manage_state(self):
        pass

    @abstractmethod
    def run_state_machine(self):
        pass
    
    def save_hist(self, run_name):
        filepath = f"flights_data/{run_name}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self.history, f)
            print(f"Flight {run_name} saved")

