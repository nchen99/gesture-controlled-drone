import openpose_tf as op
import cv2
op.init(op.MODEL_PATH)

# path_to_vid
path_to_vid = "./assets/testvid.mp4"

cap = cv2.VideoCapture(path_to_vid)
 
while(cap.isOpened()):
    ret, frame = cap.read()

    # This condition prevents from infinite looping
    # incase video ends.
    if ret == False:
        break
    frame.flags.writeable = False
    print(op.get_pose(frame))
 
cap.release()
cv2.destroyAllWindows()
