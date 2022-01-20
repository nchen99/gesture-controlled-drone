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
import sys
from PIL import Image, ImageDraw, ImageFont
from model_processors.BaseProcessor import BaseProcessor

from atlas_utils.acl_dvpp import Dvpp
from atlas_utils.acl_image import AclImage


class ModelProcessor(BaseProcessor):

    gesture_categories = [
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        'left',
        'ok',
        'right',
        'rock',
        'finger heart',
        'praise',
        'prayer',
        'stop',
        'Give the middle finger',
        'bow',
        'No gesture'
    ]

    def __init__(self, params):
        super().__init__(params)
        self._dvpp = Dvpp(self._acl_resource)
        if not os.path.exists("../data/gesture_yuv"):
            os.mkdir("../data/gesture_yuv")
        self._tmp_file = "../data/gesture_yuv/tmp.jpg"

    """Try with default ACLImage and DVPP implementation - then implement CV2 and image memory implement if time permits"""
    def preprocess(self, image):
        image_dvpp = image.copy_to_dvpp()
        yuv_image = self._dvpp.jpegd(image_dvpp)
        resized_image = self._dvpp.resize(yuv_image, self._model_width, self._model_height)
        return resized_image

    def postprocess(self, infer_output, origin_img):
        data = infer_output[0]
        vals = data.flatten()
        top_k = vals.argsort()[-1:-2:-1]
        if len(top_k):
            object_class = self.get_gesture_categories(top_k[0])
            origin_img = Image.fromarray(origin_img)
            draw = ImageDraw.Draw(origin_img)
            font = ImageFont.load_default()
            draw.text((10, 50), object_class, font=font, fill=255)
            return np.array(origin_img), object_class

        return np.array(origin_img), "No gesture"
    
    def predict(self, frame):
        cv2.imwrite(self._tmp_file, frame)
        self._acl_image = AclImage(self._tmp_file)
        resized_image = self.preprocess(self._acl_image)
        infer_out = self.model.execute([resized_image,])
        result, command = self.postprocess(infer_out, frame)
        return result, command

    def get_gesture_categories(self, gesture_id):
        if gesture_id >= len(ModelProcessor.gesture_categories):
            return "unknown"
        else:
            return ModelProcessor.gesture_categories[gesture_id]

    
