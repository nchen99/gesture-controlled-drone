import sys
import cv2
import numpy as np

sys.path.append("..")

from inference_filters.DecisionFilter import DecisionFilter
from model_processors.FaceDetectionProcessor import sigmoid, yolo_head, yolo_correct_boxes, yolo_boxes_and_scores, nms, yolo_eval, get_box_img
from TelloPIDController import TelloPIDController
import openpose_tf
import math
import time

class PIDOpenPoseTracker(TelloPIDController):
    """
    Closed-Loop Face Detection + Tracking System
    Control relies on feedback from in a closed-loop manner - enable drone to automatically adjust itself without user intervention to detect and track a 
    person's face

    :class:
        PIDTracker uses proportional integral derivative to enable continuous modulated control inherited from TelloPIDController class
    :params:
        + pid               - List(Floats); A list of three floating values that corresponds to proportional, integral and derivative
        + inference_filter  - Class:Filter; Inference result filter to smooth the model's inference results
        + if_fps            - Int; Argument for inference_filter class, rate of incomimg frames
        + if_window         - Int; Argument for inference_filter class, period of observation (in seconds) 
        + save_flight_hist  - Bool; Save flight statistics if True
    Returns
        None  
    """
    def __init__(self, pid, inference_filter=DecisionFilter, save_flight_hist=False):
        super().__init__(pid, save_flight_hist)
        openpose_tf.init(openpose_tf.MODEL_PATH)
        self.inference_filter = inference_filter    # A fully instantiated InferenceFilter object  
        # ------------------------------- Don't forget to change the value here------------
        print("PIDOpenPoseTracker.py line 34, don't forget to change the variables")
        self.setpoint_area = (80, 120)        # Lower and Upper bound for Forward&Backward Range-of-Motion - can be adjusted    
        # ------------------------------- Don't forget to change the value here------------
        self.save_flight_hist = save_flight_hist
        self.nose = None
        self.neck = None

           
    def _unpack_feedback(self, frame):
        """ Extract Process Variables from model's inference output info of input frame. The largest bbox of the same ToI label will be marked as ToI
        :params:
            infer_output - model's inference result from executing model on a frame
            frame        - input frame for inference
            toi          - Target-of-Interest 
        Returns
            process_var_center  - Process Variable - ToI's bbox center
            process_var_bbox    - Process Variable - ToI's bbox area
            result_img   - inference result superimposed on original frame
        """
        
        return openpose_tf.get_bounding_box(frame)
        # process_var_bbox_area = 0
        # process_var_bbox_center = None
        # frame = openpose_tf.get_bounding_box(frame)

        # return frame, (process_var_bbox_area, process_var_bbox_center)


    def _pid_controller(self, result, prev_x_err, prev_y_err):
        """Closed-Loop PID Object Tracker (Compensator + Actuator)
        Calculates the Error value from Process variables and compute the require adjustment for the drone. 

        Process Variable Area - for calculating the distance between drone and ToI; Info obtained from inference bbox; adjusts forward and backward motion of drone
        Process Variable center - for calculating how much to adjust the camera angle and altitude of drone
                    x_err: YAW angle rotation
                    y_err: elevation to eye-level (NotImplemented)
        :params:
            + process_vars - Tuple(process variables bbox area and process variable bbox center from unpacking feedback)
            + prev_x_err   - x error from previous control loop
            + prev_y_err   - y error from previous control loop
        Returns
            x_err, y_err   - current control loop error 
        """
        dist, center = result["dist"], result["center"]

        if dist == 0 and center is None:
            return prev_x_err, prev_y_err
        
        x_err = center[0] - self.setpoint_center[0]       # rotational err
        y_err = self.setpoint_center[1] - center[1]       # elevation err

        # Velocity signals for the drone
        forward_backward_velocity = 0
        left_right_velocity = 0
        up_down_velocity = 0
        yaw_velocity = 0


        # Compensator: calcuate the amount adjustment needed in YAW axis and distance
        # Localization of ToI to the center x-axis - adjusts camera angle
        if x_err != 0:
            yaw_velocity = 1.2 * self._pid(x_err, prev_x_err)
            yaw_velocity = int(np.clip(yaw_velocity, -120, 120))

        

        # Localization of ToI to the center y-axis - adjust altitude 
        if y_err != 0:
            up_down_velocity = 3*self._pid(y_err, prev_y_err)
            up_down_velocity = int(np.clip(up_down_velocity, -50, 50))

        # Rectify distance between drone and target from bbox area: Adjusts forward and backward motion
        if dist > self.setpoint_area[0] and dist < self.setpoint_area[1]:
            forward_backward_velocity = 0
        elif dist < self.setpoint_area[0]:
            forward_backward_velocity = 20
        elif dist > self.setpoint_area[1]:
            forward_backward_velocity = -25


        if result["right_arm_up"]:
            left_right_velocity -= 35
            yaw_velocity += 30
        
        if result["left_arm_up"]:
            left_right_velocity += 35
            yaw_velocity -= 30

        # Saves run history in a list for serialization
        if self.save_flight_hist:
            history = {
                "left_right_velocity": left_right_velocity,
                "forward_backward_velocity": forward_backward_velocity, 
                "up_down_velocity": up_down_velocity,
                "yaw_velocity": yaw_velocity,
                "x_err": x_err,
                "y_err": y_err,
                "pv_bbox_area": dist,
                "pv_center": center
            }
            self.history.append(history)

        # Actuator - adjust drone's motion to converge to setpoint by sending rc control commands
        self.uav.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)
        return x_err, y_err

    def _track(self, result, prev_x_err, prev_y_err):
        x_err, y_err = self._pid_controller(result, prev_x_err, prev_y_err)
        return x_err, y_err

    def _search(self):
        """Send RC Controls to drone to try to find ToI"""
        self.uav.send_rc_control(0,0,0,20)
        return



    def _manage_state(self, frame):
        """State Manager
        Infer surroundings to check if ToI is present, pass feedback to Filter to smooth out detection result. Break out of Search Mode 
        and enter Track Mode if ToI is consistently present. Vice versa.
        :params:
            + frame     - input frame from video stream
            + toi       - Target-of-Interest, defaults to Person for Person detection
        Returns
            result_img   - inference result superimposed on frame
            process_vars - Tuple(bbox_area, bbox_center) of process variables
        """
        results = self._unpack_feedback(frame)
        if len(results) == 0:
            return None, None

        result = results[0]
        if self.nose == None or self.neck == None:
            # search someone with the largest dist:
            result = max(results, key = lambda i : i["dist"])

        else:
            calc_dist = lambda p1, p2 : math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            max_dist = lambda i : max(calc_dist(i["nose"], self.nose), calc_dist(i["neck"], self.neck))
            # search with lowest difference:
            result = min(results, key=max_dist)
            
        self.nose = result["nose"]
        self.neck = result["neck"]
        
        # ----------------------------------- area is not used here!!! -------------------------
        # dist, center = result["dist"], result["center"]s
        center = result["center"]

        cur_mode = "TRACK" if self.track_mode else "SEARCH"
        sample_val = center if center is None else "Presence"
        filtered_result = self.inference_filter.sample(sample_val)

        if filtered_result == "MODE_INFERENCE_SAMPLING":
            pass
        elif filtered_result == "Presence": 
            if not self.track_mode and self.search_mode:
                # self.uav.move_up(50)
                # self.uav.move_down(50)
                self.uav.flip_left()
            self.track_mode = True
            self.search_mode = False
        elif filtered_result is None:
            self.track_mode = False
            self.search_mode = True
        
        new_mode =  "TRACK" if self.track_mode else "SEARCH"
        if cur_mode != new_mode: 
            print("\n######################################################")
            print(f"Mode switched from {cur_mode} to {new_mode}")

        return frame, result
    
    def run_state_machine(self, frame, prev_x_err, prev_y_err):
        result_img, result = self._manage_state(frame)
        if self.search_mode or result is None:
            self._search()
            return prev_x_err, prev_y_err, result_img
        elif self.track_mode:
            x_err, y_err = self._track(result, prev_x_err, prev_y_err)
            return x_err, y_err, result_img
    
    def __repr__(self):
        return f"PIDFaceTracker(pid={self.pid}, inference_filter={self.inference_filter})"