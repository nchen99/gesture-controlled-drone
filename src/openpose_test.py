import openpose_tf as op
import cv2 as cv
op.init(op.MODEL_PATH)
result1 = op.get_pose(cv.imread("./assets/in.png"))
result2 = op.get_pose(cv.imread("./assets/in2.jpg"))
result3 = op.get_pose(cv.imread("./assets/in3.jpg"))
print(result1, result2, result3)
