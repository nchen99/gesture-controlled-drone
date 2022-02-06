from djitellopy import Tello
import openpose_tf

if __name__ == "__main__":
    tello = Tello()
    tello.connect()
    print(tello.get_battery(), "\n\n\n")
    tello.streamon()
    frame= tello.get_frame_read().frame
    print(frame)

    openpose_tf.init()

    while True:

        _, h = openpose_tf.get_bounding_box(tello.get_frame_read().frame)
        area = h[0]
        print(area)

    
