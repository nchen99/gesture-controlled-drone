# gesture-controlled-drone
UBC ECE Capstone Project Team PN-17 (University of Bad Coders)

## HiFly Link
https://github.com/Ascend-Huawei/HiFly_Drone

## Final Demo Video
https://www.youtube.com/watch?v=7ZVxNtVQn8Q

## Requirements 
1. Hardware: Atlas 200DK board, DJI Tello drone, and a router
2. Ensure your Atlas 200DK board follows the setup steps found in https://www.notion.so/Atlas-200-DK-Setup-Guide-070b907c3c124381bdd6721618b81ef8, as well as all the requirements in this repository's requirements.txt 
3. Clone this repository into the Atlas 200DK board, make sure you are on the main branch

## Project Setup Steps
1. Power on Atlas 200DK board and connect it to Windows PC with USB. 
2. Power on the Router and connect it to Atlas 200DK board using Ethernet Cable.
3. Power on DJI Tello drone.
4. Config static IP for USB RNDIS Adapter
5. On your Windows PC, go to Advanced network settings -> Change adaptor options and find USB RNDIS Adapter.
6. Right click on USB RNDIS Adapter and select Properties.
7. Double click Internet Protocol Version 4 (TCP/IPv4), select Use the following IP address, change IP address to 192.168.1.223, change Subnet mask to 255.255.255.0
8. Enable Network sharing through USB
9. On your Windows PC, go to Advanced network settings -> Change adaptor options and find Your Wifi or Ethernet Apapter.
10. Right click on Your Wifi or Ethernet Apapter and select Properties.
11. Select sharing tab, check Allow other network users to connect through this computer’s internet connection, and pick the network name of the USB RNDIS Adapter.
12. Repeat step 7 if needed.
13. Login to the board using ssh HwHiAiUser@192.168.1.2 with password Mind@123
14. Double check that the board is able to access the internet using commands such as curl google.ca

### To run the integrated project: 
1. Go to directory  ~/src/pid_controllers/, run python3 run_track.py to start the project


## State Workflow
![state_workflow.jpg](https://github.com/nchen99/gesture-controlled-drone/blob/ea75b557a639cf6aa32190e1be4fd2f3d2ada076/state_workflow.jpg?raw=true)

## Our Implementation

### Gesture Recognition

### Face Tracking
With OpenPose(https://github.com/CMU-Perceptual-Computing-Lab/openpose), we can recognize the nose and neck of a person. We draw a line between the nose and neck and we measure the distance (in pixel). If the distance is too large, it means the drone is too close to the person and it should move backward; if the distance is too small, it means the drone is too far away from the person and it should move forward. 

If there are multiple people in one frame, it's important for the drone to "always follow one person". We define D(p1, p2) (aka the distance between person 1 and person 2) to be max(D(p1.nose, p2.nose), D(p1.neck, p2.neck)), where D(p1.nose, p2.nose) means the Euclidean distance in pixel between the nose of the first person and the nose of the second person. In the next frame, we calculate the D(p', p) for each person p' (person p is the person that the drone tracked in the last frame), and pick up the person that has the smallest D(p', p) to track.

[![Screen-Shot-2022-04-25-at-10-55-11-AM.png](https://i.postimg.cc/Nfnr1Dpk/Screen-Shot-2022-04-25-at-10-55-11-AM.png)](https://postimg.cc/SJWs460n)

### Obstacle Avoidance(Future Improvement)
We added 6 infrared sensors in 6 directions, each sensor will tell the drone how far away is an abstacle in a direction. The 200-DK board will then figure out in which direction and in what speed the drone should move away from the obstacle. Here is a link to a video demo of the tests completed so far: https://youtu.be/gh6r0lO72FQ

[![Screen-Shot-2022-04-25-at-11-00-57-AM.png](https://i.postimg.cc/KjMhCZhz/Screen-Shot-2022-04-25-at-11-00-57-AM.png)](https://postimg.cc/SYkPMbsF)

