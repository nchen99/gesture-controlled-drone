from djitellopy import Tello

tello = Tello()
tello.connect()
tello.streamon()

print(tello.get_battery())
obj = tello.get_frame_read()
print(obj.frame)
