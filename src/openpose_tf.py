
"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import cv2
import numpy as np
import argparse
import sys
import time
import math
from enum import Enum
# import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

import tf_pose.pafprocess.pafprocess as pafprocess

from atlas_utils.acl_resource import AclResource as AclLiteResource
from atlas_utils.acl_model import Model as AclLiteModel


MODEL_PATH = os.path.join(
    "/home/HwHiAiUser/CPEN491/model/OpenPose_for_TensorFlow_BatchSize_1.om")
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "./assets/in.png"

print("MODEL_PATH:", MODEL_PATH)


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()


def _include_part(part_list, part_idx):
    for part in part_list:
        if part_idx == part.part_idx:
            return True, part
    return False, None


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _round(v):
        return int(round(v))

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def get_face_box(self, img_w, img_h, mode=0):
        """
        Get Face box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :param mode:
        :return:
        """
        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _REye = CocoPart.REye.value
        _LEye = CocoPart.LEye.value
        _REar = CocoPart.REar.value
        _LEar = CocoPart.LEar.value

        _THRESHOLD_PART_CONFIDENCE = 0.2
        parts = [part for idx, part in self.body_parts.items(
        ) if part.score > _THRESHOLD_PART_CONFIDENCE]

        is_nose, part_nose = _include_part(parts, _NOSE)
        if not is_nose:
            return None

        size = 0
        is_neck, part_neck = _include_part(parts, _NECK)
        if is_neck:
            size = max(size, img_h * (part_neck.y - part_nose.y) * 0.8)

        is_reye, part_reye = _include_part(parts, _REye)
        is_leye, part_leye = _include_part(parts, _LEye)
        if is_reye and is_leye:
            size = max(size, img_w * (part_reye.x - part_leye.x) * 2.0)
            size = max(size,
                       img_w * math.sqrt((part_reye.x - part_leye.x) ** 2 + (part_reye.y - part_leye.y) ** 2) * 2.0)

        if mode == 1:
            if not is_reye and not is_leye:
                return None

        is_rear, part_rear = _include_part(parts, _REar)
        is_lear, part_lear = _include_part(parts, _LEar)
        if is_rear and is_lear:
            size = max(size, img_w * (part_rear.x - part_lear.x) * 1.6)

        if size <= 0:
            return None

        if not is_reye and is_leye:
            x = part_nose.x * img_w - (size // 3 * 2)
        elif is_reye and not is_leye:
            x = part_nose.x * img_w - (size // 3)
        else:
            # is_reye and is_leye:
            x = part_nose.x * img_w - size // 2

        x2 = x + size
        if mode == 0:
            y = part_nose.y * img_h - size // 3
        else:
            y = part_nose.y * img_h - self._round(size / 2 * 1.2)
        y2 = y + size

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if self._round(x2 - x) == 0.0 or self._round(y2 - y) == 0.0:
            return None
        if mode == 0:
            return {"x": self._round((x + x2) / 2),
                    "y": self._round((y + y2) / 2),
                    "w": self._round(x2 - x),
                    "h": self._round(y2 - y)}
        else:
            return {"x": self._round(x),
                    "y": self._round(y),
                    "w": self._round(x2 - x),
                    "h": self._round(y2 - y)}

    def get_upper_body_box(self, img_w, img_h):
        """
        Get Upper body box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :return:
        """

        if not (img_w > 0 and img_h > 0):
            raise Exception("img size should be positive")

        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _RSHOULDER = CocoPart.RShoulder.value
        _LSHOULDER = CocoPart.LShoulder.value
        _THRESHOLD_PART_CONFIDENCE = 0.3
        parts = [part for idx, part in self.body_parts.items(
        ) if part.score > _THRESHOLD_PART_CONFIDENCE]
        part_coords = [(img_w * part.x, img_h * part.y) for part in parts if
                       part.part_idx in [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]]

        if len(part_coords) < 5:
            return None

        # Initial Bounding Box
        x = min([part[0] for part in part_coords])
        y = min([part[1] for part in part_coords])
        x2 = max([part[0] for part in part_coords])
        y2 = max([part[1] for part in part_coords])

        # # ------ Adjust heuristically +
        # if face points are detcted, adjust y value

        is_nose, part_nose = _include_part(parts, _NOSE)
        is_neck, part_neck = _include_part(parts, _NECK)
        if is_nose and is_neck:
            y -= (part_neck.y * img_h - y) * 0.8

        # # by using shoulder position, adjust width
        is_rshoulder, part_rshoulder = _include_part(parts, _RSHOULDER)
        is_lshoulder, part_lshoulder = _include_part(parts, _LSHOULDER)
        if is_rshoulder and is_lshoulder:
            half_w = x2 - x
            dx = half_w * 0.15
            x -= dx
            x2 += dx
        elif is_neck:
            if is_lshoulder and not is_rshoulder:
                half_w = abs(part_lshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)
            elif not is_lshoulder and is_rshoulder:
                half_w = abs(part_rshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)

        # ------ Adjust heuristically -
        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if self._round(x2 - x) == 0.0 or self._round(y2 - y) == 0.0:
            return None
        return {"x": self._round((x + x2) / 2),
                "y": self._round((y + y2) / 2),
                "w": self._round(x2 - x),
                "h": self._round(y2 - y)}

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()


def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling
    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape + kernel_size,
                     strides=(stride*A.strides[0],
                              stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


def nms(heatmaps):
    results = np.empty_like(heatmaps)
    for i in range(heatmaps.shape[-1]):
        heat = heatmaps[:, :, i]
        hmax = pool2d(heat, 3, 1, 1)
        keep = (hmax == heat).astype(float)

        results[:, :, i] = heat * keep
    return results


def estimate_paf(peaks, heat_mat, paf_mat):
    pafprocess.process_paf(peaks, heat_mat, paf_mat)

    humans = []
    for human_id in range(pafprocess.get_num_humans()):
        human = Human([])
        is_added = False

        for part_idx in range(18):
            c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
            if c_idx < 0:
                continue

            is_added = True
            human.body_parts[part_idx] = BodyPart(
                '%d-%d' % (human_id, part_idx), part_idx,
                float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
                float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0],
                pafprocess.get_part_score(c_idx)
            )

        if is_added:
            score = pafprocess.get_score(human_id)
            human.score = score
            humans.append(human)

    return humans


def draw(img, humans):
    # img = cv2.resize(img, None, fx=1/10, fy=1/10, interpolation=cv2.INTER_AREA)
    edges = [
        (CocoPart.Nose.value, CocoPart.Neck.value),
        (CocoPart.Nose.value, CocoPart.LEye.value), (CocoPart.LEye.value,
                                                     CocoPart.LEar.value),
        (CocoPart.Nose.value, CocoPart.REye.value), (CocoPart.REye.value,
                                                     CocoPart.REar.value),
        (CocoPart.Neck.value, CocoPart.LShoulder.value), (CocoPart.Neck.value,
                                                          CocoPart.RShoulder.value),
        (CocoPart.LShoulder.value,
         CocoPart.LElbow.value), (CocoPart.RShoulder.value, CocoPart.RElbow.value),
        (CocoPart.LElbow.value, CocoPart.LWrist.value), (CocoPart.RElbow.value,
                                                         CocoPart.RWrist.value),
        (CocoPart.Neck.value, CocoPart.LHip.value), (CocoPart.Neck.value,
                                                     CocoPart.RHip.value),
        (CocoPart.LHip.value, CocoPart.LKnee.value), (CocoPart.RHip.value,
                                                      CocoPart.RKnee.value),
        (CocoPart.LKnee.value, CocoPart.LAnkle.value), (CocoPart.RKnee.value,
                                                        CocoPart.RAnkle.value)
    ]
    h, w, _ = img.shape
    for human in humans:
        color = list(np.random.random(size=3) * 256)
        for key, body_part in human.body_parts.items():
            center = (int(w*body_part.x), int(h*body_part.y))
            cv2.circle(img, center, radius=4, thickness=-1, color=color)

        for edge in edges:
            if set(edge) <= set(human.body_parts):
                p1, p2 = edge
                start = (int(w*human.body_parts[p1].x),
                         int(h*human.body_parts[p1].y))
                end = (int(w*human.body_parts[p2].x),
                       int(h*human.body_parts[p2].y))
                cv2.line(img, start, end, color=color, thickness=2)

    pass
    filename = os.path.basename(f"{time.time_ns()}.png")
    cv2.imwrite(f"outputs/{filename}", img)


def post_process(heat):
    heatMat = heat[:, :, :19]
    pafMat = heat[:, :, 19:]

    ''' Visualize Heatmap '''
    # print(heatMat.shape, pafMat.shape)
    # for i in range(19):
    #     plt.imshow(heatMat[:,:,i])
    # plt.savefig("outputs/heatMat.png")

    # blur = cv2.GaussianBlur(heatMat, (25, 25), 3)

    peaks = nms(heatMat)
    humans = estimate_paf(peaks, heatMat, pafMat)
    return humans


def pre_process(img, height=368, width=656):
    model_input = cv2.resize(img, (width, height))

    return model_input[None].astype(np.float32).copy()


acl_resource = None
model = None


class Pose(Enum):
    KEY_MISSING = -1
    NONE = 0  # any
    RIGHT_ARM_UP = 1  # right arm up
    CONFIRM = 2  # right hand to shoulder
    BOTH_ARM_UP = 3  # double hand up
    CLAP = 4  # hand together
    LEFT_ARM_UP = 5  # left hand up


threshold = 0.01

check_pose = {
    Pose.CONFIRM: [
        {
            "req": [CocoPart.RWrist.value, CocoPart.RShoulder.value],
            "check": (lambda bp: (bp[CocoPart.RWrist.value].x - bp[CocoPart.RShoulder.value].x) ** 2 + (bp[CocoPart.RWrist.value].y - bp[CocoPart.RShoulder.value].y) ** 2 < threshold)
        },
    ],
    Pose.CLAP: [
        {
            "req": [CocoPart.RWrist.value, CocoPart.LWrist.value],
            "check": (lambda bp: (bp[CocoPart.RWrist.value].x - bp[CocoPart.LWrist.value].x) ** 2 + (bp[CocoPart.RWrist.value].y - bp[CocoPart.LWrist.value].y) ** 2 < threshold)
        }
    ],
    Pose.BOTH_ARM_UP: [
        {
            "req": [CocoPart.RWrist.value, CocoPart.RShoulder.value, CocoPart.LWrist.value, CocoPart.LShoulder.value],
            "check": (lambda bp: bp[CocoPart.RWrist.value].y < bp[CocoPart.RShoulder.value].y and bp[CocoPart.LWrist.value].y < bp[CocoPart.LShoulder.value].y)
        },
        {
            "req": [CocoPart.RElbow.value, CocoPart.RShoulder.value, CocoPart.LElbow.value, CocoPart.LShoulder.value],
            "check": (lambda bp: bp[CocoPart.RElbow.value].y < bp[CocoPart.RShoulder.value].y and bp[CocoPart.LElbow.value].y < bp[CocoPart.LShoulder.value].y)
        }
    ],
    Pose.RIGHT_ARM_UP: [
        {
            "req": [CocoPart.RWrist.value, CocoPart.RShoulder.value, CocoPart.LWrist.value, CocoPart.LShoulder.value],
            "check": (lambda bp: bp[CocoPart.RWrist.value].y < bp[CocoPart.RShoulder.value].y and bp[CocoPart.LWrist.value].y > bp[CocoPart.LShoulder.value].y)
        },
        {
            "req": [CocoPart.RElbow.value, CocoPart.RShoulder.value, CocoPart.LElbow.value, CocoPart.LShoulder.value],
            "check": (lambda bp: bp[CocoPart.RElbow.value].y < bp[CocoPart.RShoulder.value].y and bp[CocoPart.LElbow.value].y > bp[CocoPart.LShoulder.value].y)
        }
    ],
    Pose.LEFT_ARM_UP: [
        {
            "req": [CocoPart.RWrist.value, CocoPart.RShoulder.value, CocoPart.LWrist.value, CocoPart.LShoulder.value],
            "check": (lambda bp: bp[CocoPart.RWrist.value].y > bp[CocoPart.RShoulder.value].y and bp[CocoPart.LWrist.value].y < bp[CocoPart.LShoulder.value].y)
        },
        {
            "req": [CocoPart.RElbow.value, CocoPart.RShoulder.value, CocoPart.LElbow.value, CocoPart.LShoulder.value],
            "check": (lambda bp: bp[CocoPart.RElbow.value].y > bp[CocoPart.RShoulder.value].y and bp[CocoPart.LElbow.value].y < bp[CocoPart.LShoulder.value].y)
        }
    ],
    Pose.NONE: [
        {
            "req": [],
            "check": (lambda bp: True)
        }
    ]
}

boxReq = [CocoPart.Nose.value,
          CocoPart.RShoulder.value, CocoPart.LShoulder.value, CocoPart.Neck.value]


def analyze_pose(human):
    global threshold
    # print(human)

    bp = human.body_parts

    for pose in check_pose.keys():
        for entry in check_pose[pose]:
            if set(entry["req"]) <= set(bp):
                if entry["check"](bp):
                    return pose

    return Pose.NONE
    # try:
    #     if (bp[CocoPart.RWrist.value].x - bp[CocoPart.RShoulder.value].x) ** 2 + (bp[CocoPart.RWrist.value].y - bp[CocoPart.RShoulder.value].y) ** 2 < threshold:
    #         return Pose.CONFIRM
    #
    #     if (bp[CocoPart.RWrist.value].x - bp[CocoPart.LWrist.value].x) ** 2 + (bp[CocoPart.RWrist.value].y - bp[CocoPart.LWrist.value].y) ** 2 < threshold:
    #         return Pose.CLAP
    #
    #
    #     # Raising hands
    #     if bp[CocoPart.RWrist.value].y < bp[CocoPart.RShoulder.value].y:
    #         if bp[CocoPart.LWrist.value].y < bp[CocoPart.LShoulder.value].y:
    #             return Pose.BOTH_ARM_UP
    #         else:
    #             return Pose.RIGHT_ARM_UP
    #     else:
    #         if bp[CocoPart.LWrist.value].y < bp[CocoPart.LShoulder.value].y:
    #             return Pose.LEFT_ARM_UP
    # except:
    #    return Pose.KEY_MISSING
    #
    # return Pose.NONE


def init(model_path):
    print("hahahahah\n\n\n\n\n\n", model_path)
    global acl_resource, model
    acl_resource = AclLiteResource()
    acl_resource.init()

    model = AclLiteModel(model_path)


# input: cv2 Mat
def get_pose(img):
    global model
    model_input = pre_process(img)
    output = model.execute([model_input])
    humans = post_process(output[0][0])
    results = []
    for human in humans:
        results.append(analyze_pose(human))

    draw(img, humans)
    # also include the cordinates? => can keep track of the target when there are multiple humans
    return results

# returns an array of four coordinates to be used as bounding box




def get_bounding_box(img):
    global model
    model_input = pre_process(img)
    # print("model_input: ", model_input)
    # print("model_input length: ", len(model_input))
    output = model.execute([model_input])
    humans = post_process(output[0][0])
    results = []
    # draw(img, humans)
    for human in humans:
        # processing here
        result = calculate_bounding_box(human, img.shape[0], img.shape[1])
        if result is None:
            continue
        results.append(result)
        

    for result in results:
        draw_box(result, img)

    return results

def check_pose_func(pose, human):
    global threshold

    bp = human.body_parts

    for entry in pose:
        if set(entry["req"]) <= set(bp):
            if entry["check"](bp):
                return True

    return False


def calculate_bounding_box(human, h, w):
    # logic: check for identifiable points
    # case 1: person is facing forward, all facial points are available
    # => use nose as center point, shoulder distance as width, shoulder to nose distance as height
    # case 2: person is facing sideways or backwards, in which we can only detect the neck, maybe a shoulder or two
    # => neck = center of box, draw based on preset radius
    # => alternatively return inconclusive and make the drone go into search mode until a box can be found
    bp = human.body_parts
    boxCoordinates = []
    # print(bp)

    # case1 => CocoPart.Nose.value, CocoPart.RShoulder.value, CocoPart.LShoulder.value
    if(set(boxReq) <= set(bp)):
        # center
        x = bp[CocoPart.Nose.value].x
        y = bp[CocoPart.Nose.value].y
        dx = abs(bp[CocoPart.Nose.value].x - bp[CocoPart.RShoulder.value].x)
        dy = abs(bp[CocoPart.Nose.value].y - bp[CocoPart.RShoulder.value].y)

        convert = lambda i : (int(i.x * w), int(i.y * h))

        nose = convert(bp[CocoPart.Nose.value])
        neck = convert(bp[CocoPart.Neck.value])


        # return boxCoordinates, (400000*dx*dy, [x, y])
        return {
            "ur": [int(w * (x + dx)), int(h * (y + dy))],
            "ul": [int(w * (x - dx)), int(h * (y + dy))],
            "ll": [int(w * (x - dx)), int(h * (y - dy))],
            "lr": [int(w * (x + dx)), int(h * (y - dy))],
            "center": [int(w * x), int(h * y)],
            "area": int(2 * dx * w) * int(2 * dy * h),
            "nose": nose,
            "neck": neck,
            "dist": int(math.sqrt((nose[0] - neck[0]) ** 2 + (nose[1] - neck[1]) ** 2)),
            "right_arm_up": check_pose_func(check_pose[Pose.RIGHT_ARM_UP], human),
            "left_arm_up": check_pose_func(check_pose[Pose.LEFT_ARM_UP], human),
            "clap": check_pose_func(check_pose[Pose.CLAP], human),
            # TODO: Please add a pose for land here:
            "land": check_pose_func(check_pose[Pose.BOTH_ARM_UP], human),
            # TODO: Please add a pose for unfollow (cross sign):
            "unfollow": False
        }

    return None


def draw_box(coordinates, img):
    # print(coordinates)
    color = (0, 255, 0)
    cv2.line(img, (coordinates["ur"][0], coordinates["ur"][1]), (coordinates["ul"][0], coordinates["ul"][1]), color=color, thickness=2)
    cv2.line(img, (coordinates["ul"][0], coordinates["ul"][1]), (coordinates["ll"][0], coordinates["ll"][1]), color=color, thickness=2)
    cv2.line(img, (coordinates["ll"][0], coordinates["ll"][1]), (coordinates["lr"][0], coordinates["lr"][1]), color=color, thickness=2)
    cv2.line(img, (coordinates["lr"][0], coordinates["lr"][1]), (coordinates["ur"][0], coordinates["ur"][1]), color=color, thickness=2)
    cv2.line(img, coordinates["nose"], coordinates["neck"], color=color, thickness=2)
    # filename = os.path.basename(f"{time.time_ns()}.png")
    # cv2.imwrite(f"outputs/{filename}", img)


# def main(model_path, img_path):
#     """main"""
#     #initialize acl runtime
#     acl_resource = AclLiteResource()
#     acl_resource.init()
#
#     model = AclLiteModel(model_path)
#
#     img  = cv2.imread(img_path)
#
#     _st = time.time()
#     st = time.time()
#     model_input = pre_process(img)
#     print("\n=========================================")
#     print("pre process: ", time.time() - st); st = time.time()
#     # input1 = np.random.rand(1, 368, 656, 3).astype(np.float32)
#     output = model.execute([model_input])
#     print("inference:   ", time.time() - st); st = time.time()
#     humans = post_process(output[0][0])
#     print("post process:", time.time() - st)
#     print("total:       ", time.time() - _st)
#     print("fps:         ", 1/(time.time() - _st))
#     print("=========================================\n")
#
#     print("num humans:", len(humans))
#
#     print(humans[0])
#
#     draw(img_path, humans)

def main(model_path, img_path):
    global model
    init(model_path)
    img = cv2.imread(img_path)
    print(get_pose(img))


if __name__ == '__main__':
    description = 'Load a model for human pose estimation'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', type=str, default=MODEL_PATH)
    parser.add_argument('--img_path', type=str, default=IMAGE_PATH,
                        help="input img path e.g. /path/to/image.png")
    # parser.add_argument('--output_dir', type=str, default="outputs", help="Output Path")

    args = parser.parse_args()

    main(args.model, args.img_path)
