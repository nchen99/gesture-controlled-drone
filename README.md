# gesture-controlled-drone
UBC ECE Capstone Project Team PN-17 (University of Bad Coders)

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
12. Login to the board using ssh HwHiAiUser@192.168.1.2 with password Mind@123
13. Double check that the board is able to access the internet using commands such as curl google.ca

### To run the integrated project: 
1. Go to directory ~/src/, prepare the presenter server using bash lib/server/run_presenter_server.sh uav_presenter_server.conf
2. Go to directory  ~/src/pid_controllers/, run python3 run_track.py to start the project
