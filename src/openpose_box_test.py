import openpose_tf as op
import cv2 as cv
op.init(op.MODEL_PATH)
'''files = [
'fullbody_botharmsup.png', 'fullbody_leftarmup.png', 'fullbody_rightarmup.png', 'handstogether.png', 'leftarmup.png', 'botharmsup.png', 'fullbody_handstogether.png', 'fullbody_rightarmshoulder.png', 'halfbody_rightarmup.jpg', 'rightarmshoulder.png'
]'''

files = [
'handstogether.png', 'leftarmup.png', 'botharmsup.png', 'halfbody_rightarmup.jpg', 'rightarmshoulder.png'
]
# result1 = op.get_pose(cv.imread("./assets/in.png"))
# result2 = op.get_pose(cv.imread("./assets/in2.jpg"))
# result3 = op.get_pose(cv.imread("./assets/in3.jpg"))

for fi in files:
    print(">>>>>",op.get_bounding_box(cv.imread(f"./assets/{fi}")), "<<<<<")
