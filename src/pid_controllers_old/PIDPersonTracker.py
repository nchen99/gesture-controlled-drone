import sys
import cv2
import numpy as np

sys.path.append("..")

from DecisionFilter import DecisionFilter
from TelloPIDController import TelloPIDController

class PIDPersonTracker(TelloPIDController):
    def __init__(self, pid, inference_filter=DecisionFilter, save_flight_hist=False):
        super().__init__(pid, save_flight_hist)
        self.model_processor = self._load_mp("yolov3")
        self.inference_filter = inference_filter    # A fully instantiated InferenceFilter object  
        self.setpoint_area = (150000, 200000)        # Lower and Upper bound for Forward&Backward Range-of-Motion - can be adjusted    
        self.save_flight_hist = save_flight_hist
           
    def _unpack_feedback(self, infer_output, frame, toi="person"):
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
        box_num = infer_output[1][0, 0]
        box_info = infer_output[0].flatten()
        scale = max(frame.shape[1] / self.model_processor._model_width, frame.shape[0] / self.model_processor._model_height)
        
        # Iterate the detected boxes and look for label that matches ToI with the largest bbox area
        process_var_bbox_area = 0
        process_var_bbox_center = None

        for n in range(int(box_num)):
            ids = int(box_info[5 * int(box_num) + n])
            label = self.model_processor.labels[ids]

            if label == toi:
                score = box_info[4 * int(box_num)+n]
                top_left_x = int(box_info[0 * int(box_num)+n] * scale)
                top_left_y = int(box_info[1 * int(box_num)+n] * scale)
                bottom_right_x = int(box_info[2 * int(box_num) + n] * scale)
                bottom_right_y = int(box_info[3 * int(box_num) + n] * scale)
                cx = (top_left_x + bottom_right_x) // 2
                cy = (top_left_y + bottom_right_y) // 2
                center = (cx, cy)
                area = (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)

                if area > process_var_bbox_area:
                    process_var_bbox_area = area
                    process_var_bbox_center = center

                    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0,0,255), 2)
                    cv2.circle(frame, center, 1, (0,0,255), -1)
        
        return frame, (process_var_bbox_area, process_var_bbox_center)

    def _pid_controller(self, process_vars, prev_x_err, prev_y_err):
        """Closed-Loop PID Object Tracker (Compensator + Actuator)
        Calculates the Error value from Process variables and compute the require adjustment for the drone. 
        XY error is error between setpoint (frame center) and process_var_bbox_center
        Process Variable Area - for calculating the distance between drone and ToI (if it is over 80% of frame, then drone needs to move back)
                    Info obtained from inference bbox
                    + forward and backward motion of drone
        Process Variable center - for calculating how much to adjust the camera angle and altitude of drone
                    x_err: angle rotation
                    y_err: elevation to eye-level
        :params:
            + process_vars - Tuple(process variables bbox area and process variable bbox center)
            + prev_x_err   - x error from previous control loop
            + prev_y_err   - y error from previous control loop
        Returns
            x_err, y_err   - current control loop error 
        """
        area, center = process_vars[0], process_vars[1]

        if area == 0 and center is None:
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
            yaw_velocity = self._pid(x_err, prev_x_err)
            yaw_velocity = int(np.clip(yaw_velocity, -100, 100))

        # Localization of ToI to the center y-axis - adjust altitude 
        if y_err != 0:
            up_down_velocity = self._pid(y_err, prev_y_err)
            up_down_velocity = int(np.clip(up_down_velocity, -50, 50))

        # Rectify distance between drone and target from bbox area: Adjusts forward and backward motion
        if area > self.setpoint_area[0] and area < self.setpoint_area[1]:
            forward_backward_velocity = 0
        elif area < self.setpoint_area[0]:
            forward_backward_velocity = 20
        elif area > self.setpoint_area[1]:
            forward_backward_velocity = -20

        # Saves run history in a list for serialization
        if self.save_flight_hist:
            history = {
                "left_right_velocity": left_right_velocity,
                "forward_backward_velocity": forward_backward_velocity, 
                "up_down_velocity": up_down_velocity,
                "yaw_velocity": yaw_velocity,
                "x_err": x_err,
                "y_err": y_err,
                "pv_bbox_area": area,
                "pv_center": center
            }
            self.history.append(history)

        # Actuator - adjust drone's motion to converge to setpoint by sending rc control commands
        self.uav.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)
        return x_err, y_err

    def _track(self, process_vars, prev_x_err, prev_y_err):
        x_err, y_err = self._pid_controller(process_vars, prev_x_err, prev_y_err)
        return x_err, y_err

    def _search(self,):
        """Send RC Controls to drone to try to find ToI"""
        self.uav.send_rc_control(0,0,0,30)
        return

    def _manage_state(self, frame, toi="person"):
        """State Manager
        Infer surroundings to check if ToI is present, pass feedback to Filter to smooth out detection result. Break out of Search Mode 
        and enter Track Mode if ToI is consistently present. Vice versa.
        :params:
            + frame     - input frame from video stream
            + toi       - Target-of-Interest, defaults to Person for Person detection
        Returns
            result_img   - inference result superimposed on frame
            process_vars - Tuple() of process variables
        """
        infer_output = self._get_feedback(frame)
        result_img, process_vars = self._unpack_feedback(infer_output, frame)
        area, center = process_vars[0], process_vars[1]

        cur_mode = "TRACK" if self.track_mode else "SEARCH"
        sample_val = center if center is None else "Presence"
        filtered_result = self.inference_filter.sample(sample_val)

        if filtered_result == "MODE_INFERENCE_SAMPLING":
            pass
        elif filtered_result == "Presence": 
            self.track_mode = True
            self.search_mode = False
        elif filtered_result is None:
            self.track_mode = False
            self.search_mode = True
        
        new_mode =  "TRACK" if self.track_mode else "SEARCH"
        if cur_mode != new_mode: 
            print("\n######################################################")
            print(f"Mode switched from {cur_mode} to {new_mode}")
    
    def run_state_machine(self, frame, prev_x_err, prev_y_err):
        result_img, process_vars = self._manage_state(frame)
        if self.search_mode:
            self._search()
            return prev_x_err, prev_y_err
        if self.track_mode:
            x_err, y_err = self._track(process_vars, prev_x_err, prev_y_err)
            return x_err, y_err 

 