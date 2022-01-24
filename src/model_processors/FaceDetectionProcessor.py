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
import acl
import os
import cv2
import numpy as np
import sys
import time
from model_processors.BaseProcessor import BaseProcessor

sys.path.append("../../../../src/lib")
from atlas_utils.resource_list import resource_list


class ModelProcessor(BaseProcessor):
    def __init__(self, params):
        super().__init__(params)
        
        # parameters for preprocessing
        self.ih, self.iw = (params['camera_height'], params['camera_width'])
        self.h, self.w = params['model_height'], params['model_width']
        self.scale = min(self.w / self.iw, self.h / self.ih)
        self.nw = int(self.iw * self.scale)
        self.nh = int(self.ih * self.scale)

        # parameters for postprocessing
        self.image_shape = [params['camera_height'], params['camera_width']]
        self.model_shape = [self.h, self.w]
        self.num_classes = 1
        self.anchors = self.get_anchors()
        
    def release_acl(self):
        print("acl resource release all resource")
        resource_list.destroy()
        if self._acl_resource.stream:
            print("acl resource release stream")
            acl.rt.destroy_stream(self._acl_resource.stream)

        if self._acl_resource.context:
            print("acl resource release context")
            acl.rt.destroy_context(self._acl_resource.context)

        print("Reset acl device ", self._acl_resource.device_id)
        acl.rt.reset_device(self._acl_resource.device_id)
        acl.finalize()
        print("Release acl resource success")

    def predict(self, frame):
        preprocessed = self.preprocess(frame)
        outputs = self.model.execute([preprocessed])
        
        postprocess_start = time.process_time()
        result = self.postprocess(frame, outputs)
        print(f"@predict.postprocess = {time.process_time() - postprocess_start}")
        return result

    def preprocess(self, frame):
        """preprocess frame from drone"""
        # preprocessing: resize and paste input image to a new image with size 416*416
        img = np.array(frame, dtype='float32')
        img_resize = cv2.resize(img, (self.nw, self.nh), interpolation=cv2.INTER_CUBIC)
        img_new = np.ones((416, 416, 3), np.float32) * 128
        img_new[(self.h - self.nh) // 2: ((self.h - self.nh) // 2 + self.nh),
                (self.w - self.nw) // 2: (self.w - self.nw) // 2 + self.nw, :] = img_resize[:, :, :]
        img_new = img_new / 255.
        return img_new
        
    def postprocess(self, frame, outputs):
        yolo_eval_start = time.process_time()
        box_axis, box_score = yolo_eval(
            outputs, self.anchors, self.num_classes, self.image_shape)
        yolo_eval_end = time.process_time() - yolo_eval_start
        nparryList, boxList = get_box_img(frame, box_axis)

        if len(nparryList) > 0:
            for box in boxList:
                cv2.rectangle(frame, (box[0], box[2]),  (box[1], box[3]), (255, 0, 0), 4) 

        print(f"\n####################################################################")
        print(f"@postprocess.yolo_eval process duration = {round(yolo_eval_end, 3)}")
        # print(f"@postprocess:getbox process duration = {round(getbox_end, 3)}")
        # print(f"@postprocess:forloop process duration = {round(forloop_end, 3)}")

        return frame, yolo_eval_end
    
    def get_anchors(self):
        """return anchors

        Returns:
            [ndarray]: anchors array
        """
        anchors = np.array([[10.,13.],  [16.,30.],  [33.,23.],  [30.,61.],  [62.,45.],  [59.,119.],  [116.,90.],  [156.,198.],  [373.,326.]])
        return anchors
  
    
def sigmoid(x):
    """sigmoid"""
    x = x.astype("float32")
    return 1 / (1 + np.exp(-x))


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""

    feats = feats.astype("float32")
    num_anchors = len(anchors)  # 3
    anchors_tensor = np.reshape(anchors, [1, 1, 1, num_anchors, 2])

    grid_shape = np.shape(feats)[1:3]

    grid_y = np.tile(np.reshape(
        range(0, grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = np.tile(np.reshape(
        range(0, grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])

    grid = np.concatenate([grid_x, grid_y], axis=-1)
    grid = grid.astype("float32")

    feats = np.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    box_xy = (sigmoid(feats[..., :2]) + grid) / \
        np.cast[feats.dtype](grid_shape[::-1])
    box_wh = np.exp(feats[..., 2:4]) * anchors_tensor / \
        np.cast[feats.dtype](input_shape[::-1])

    box_confidence = sigmoid(feats[..., 4:5])
    box_class_probs = sigmoid(feats[..., 5:])

    box_wh = box_wh.astype("float32")

    ret = {
        "box_xy": box_xy,
        "box_wh": box_wh,
    }
    if calc_loss == True:
        ret["grid"] = grid
        ret["feats"] = feats
    else:
        ret["box_confidence"] = box_confidence
        ret["box_class_probs"] = box_class_probs
    return ret


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """Get corrected boxes"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = np.cast[box_yx.dtype](input_shape)
    image_shape = np.cast[box_yx.dtype](image_shape)
    new_shape = np.round(image_shape * np.min(input_shape / image_shape))

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)

    boxes = np.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)

    # Scale boxes back to original image shape.
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)

    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """Process Conv layer output"""
    heads = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(heads['box_xy'], heads['box_wh'], input_shape, image_shape)
    boxes = np.reshape(boxes, [-1, 4])
    box_scores = heads['box_confidence'] * heads['box_class_probs']
    box_scores = np.reshape(box_scores, [-1, num_classes])

    return (boxes, box_scores)


def nms(bounding_boxes, confidence_score, threshold):
    """non maximum suppression for filter boxes"""
    print(f"\n@predict.postprocess.yolo_eval.nms analysis") 
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        print("\t@nms: returns empty list")
        return [], []

    def_var_start = time.process_time()
    
    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)[::-1]
    keep = []

    def_var_end = time.process_time() - def_var_start
    print(f"\t@nms: define variables (bbox, area, etc duration: {def_var_end}")

    while_loop_start = time.process_time()
    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[0]
        keep.append(index)

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[1:]] - intersection)
        inds = np.where(ratio <= threshold)[0]
        order = order[inds + 1]

    while_loop_end = time.process_time() - while_loop_start
    print(f"\t@nms: whileloop bbox iteration: {while_loop_end}")

    picked_boxes = [bounding_boxes[i] for i in keep]
    if not score.shape:
        picked_score = [score]
    else:
        picked_score = [score[i] for i in keep]

    return picked_boxes, picked_score


def yolo_eval(yolo_outputs, anchors, num_classes, image_shape, score_threshold=.5, iou_threshold=.45):
    """
    Obtain predicted boxes axis and corresponding scores

    Args:
        yolo_outputs: output (3 feature maps) of YOLO V3 model, sizes are 1*13*13*18; 1*26*26*18; 1*52*52*18 seperately
        anchors: anchors pre-calculated
        num_classes: only 1 class here, which is "head"
        image_shape: original image input

    Returns:
        predicted boxes axis and corresponding scores
    """
    print("\n@postprocess:yolo_eval analysis:")

    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    yolo_output_0 = yolo_outputs[0]
    input_shape = [yolo_output_0.shape[1] * 32, yolo_output_0.shape[2] * 32]
    input_shape = np.array(input_shape)
    boxes = []
    box_scores = []

    # forloop start
    num_layer_start = time.process_time()
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    num_layer_end = time.process_time() - num_layer_start
    print(f"\t@yolo_eval:forloop process duration: {num_layer_end}")

    boxes = np.concatenate(boxes, axis=0)
    box_scores = np.concatenate(box_scores, axis=0)
    mask = box_scores >= score_threshold
    class_boxes = boxes[np.nonzero(box_scores * mask)[0], :]
    class_box_scores = box_scores[np.nonzero(box_scores * mask)[0], :]
    class_box_scores = np.squeeze(class_box_scores)

    # nms
    nms_start = time.process_time()
    box, score = nms(class_boxes, class_box_scores, iou_threshold)
    nms_end = time.process_time() - num_layer_start
    print(f"\t@yolo_eval:nms process duration: {nms_end}")


    return box, score


def get_box_img(image, box_axis):
    """
    Pack detected head area and corresponding location in the source image for WHENet

    Args:
        image: source image read from camera
        box_axis: location of boxes detected in YOLOV3
        box_score: scores of boxes detected in YOLOV3

    Returns:
        nparryList: head area 
        boxList: location in the source image
    """
    nparryList = []
    boxList = []

    for i in range(len(box_axis)):

        top, left, bottom, right = box_axis[i]
        top_modified = top - abs(top - bottom) / 10
        bottom_modified = bottom + abs(top - bottom) / 10
        left_modified = left - abs(left - right) / 5
        right_modified = right + abs(left - right) / 5

        top_modified = max(0, np.round(top_modified).astype('int32'))
        left_modified = max(0, np.round(left_modified).astype('int32'))
        bottom_modified = min(image.shape[0], np.round(
            bottom_modified).astype('int32'))
        right_modified = min(image.shape[1], np.round(
            right_modified).astype('int32'))

        boxList.append([left_modified, right_modified,
                       top_modified, bottom_modified])
        nparryList.append(
            image[top_modified:bottom_modified, left_modified:right_modified])

    return nparryList, boxList
