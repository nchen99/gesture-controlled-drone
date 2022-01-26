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
from abc import abstractmethod

from atlas_utils.acl_resource import AclResource
from atlas_utils.acl_model import Model

class BaseProcessor:
    def __init__(self, params):
        # Initialize ACL Resources
        self._acl_resource = AclResource()
        self._acl_resource.init()
        self.params = params
        self.validate()
        self._model_width = params['model_width']
        self._model_height = params['model_height']
        self.model = Model(params['model_path'])

    def validate(self):
        # print(self.params['model_path'])
        if not os.path.exists(self.params['model_path']):
            raise FileNotFoundError('Model Path not found, please check again.')
        if 'model_width' not in self.params or 'model_height' not in self.params:
            raise Exception('Please specify input width and height for model in params.py')

    @abstractmethod
    def preprocess(self):
        pass
        
    @abstractmethod    
    def postprocess(self):
        pass

    @abstractmethod
    def predict(self):
        pass

