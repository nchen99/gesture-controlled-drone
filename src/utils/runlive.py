
from atlas_utils.presenteragent import presenter_channel
import cv2
from utils.params import params
from atlas_utils.acl_image import AclImage
import time

class PresenterServer:
    def __init__(self, tello):
        uav_presenter_conf = params["presenter_server_conf"]
        self.chan = presenter_channel.open_channel(uav_presenter_conf)
        self.tello = tello

    def show(self, frame):
        _, jpeg_image = cv2.imencode('.jpg', frame)
        jpeg_image = AclImage(jpeg_image, frame.shape[0], frame.shape[1], jpeg_image.size)
        self.chan.send_detection_data(frame.shape[0], frame.shape[1], jpeg_image, [])

    def main(self):
        while True:
            time.sleep(0.05)
            frame = self.tello.get_frame_read().frame
            try:
                self.show(frame)
            except:
                print("bad")
                pass