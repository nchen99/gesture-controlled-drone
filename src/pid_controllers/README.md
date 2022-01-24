# PID Tello-Tracker

PID Tello-Tracker is a pseudo-tracking algorithm that is based on a PID Controller using object detection inference results from the Atlas 200 DK as continuous feedback to determine the state of the drone. 

**For more information, refer to the [PID Tracker Wiki Page](https://github.com/Ascend-Huawei/HiFly_Drone/wiki/Closed-Loop-PID-Tracker)**

### What it is:
- Written in Python 3.7.5
- Uses djitellopy wrapper to execute Tello flight commands
- Interaction between drone and tracking target is achieved via Ascend detection modules
- A simple tracker assuming trivial environment (ideally obstacles-free) and target is detectable

## Run PID Tracker
Assuming you have already gone through the main installation steps from the main [README](https://github.com/Ascend-Huawei/HiFly_Drone/tree/main) - to run this project, you will:
1. Turn on the Tello drone and connect your 200DK board to it via wireless router
2. Activate the virtual environment in the project directiory `~/HiFly_Drone` <br>
    `source venv/bin/activate`

3. Switch into the `pid_controller` directory <br>
    `cd src/pid_controller`
    
4. In the terminal, run <br>
    `python3 run_tracker.py`
    
The `run_tracker.py` accepts several optional arguments, see below table for the description of each optional argument.
|   Arguments             |         Description           |
|:-----------------------:|:-----------------------------:|
| `--flight_name`         | Name of the run of the flight for saving PID statistics. If no input, then the default flight name will be a timestamp |
| `--use_ps`              | Forwards drone video stream and inference result to Presenter Server if True. Default is False. |
| `--duration`            | Duration of flight in seconds |
| `--tracker`             | Name of the PID Tracking class to be used, i.e.: PIDFaceTracker |
| `--pid`                 | Three float values between 0 and 1 corresponding to Proportional-Integral-Derivative for the controller. Default is [0.1, 0.1, 0.1] |
| `--save_flight`         | A boolean value, if True then the flight statistics is serialized under the provided `flight_name` |
| `--inference_filter`    | Name of Inference Filter object (purpose is to smooth the inference results, thus making the PID Controller more stable). Default is DecisionFilter |
| `--if_fps`              | Specify the incoming frame rate for the InferenceFilter (int) |
| `--if_window`           | Specify the smoothing window in seconds for the InferenceFilter (int) |


> NOTE: A default `pid` value of [0.1, 0.1, 0.1] is tested and showed stable flight for face-tracking and person-tracking, but this is bound to change for other applications and manual testing is require to find the optimal. 
`if_fps` and `if_window` should also vary depends on inference rate of the detection model and responsiveness of the drone. 



## Code Description
PID Tracking functionality is enabled by the corresponding `PID<object>Tracker` class depending on the target-of-interest. `PID<object>Tracker` classes are subclass of `TelloPIDController`, 
which acts as a bridge connecting Ascend modularized capabilities to the drone while providing essential and overridable methods for its children classes. Users can therefore 
extend this application and creating their own Tracker class for other object-tracking tasks.

|   File   |         Description           |
|:--------:|:-----------------------------:|
| `run_track.py`          | Main script to run the PID Tracking application      |
| `TelloPIDController.py` | `TelloPIDController` Python class with methods to initialize the drone, load ModelProcessor detection backbone, execute model to make inference |
| `PID<object>Tracker.py` | `PID<object>Tracker` Python class, subclass of `TelloPIDController` with methods to unpack inference results and logics of PID controller|
