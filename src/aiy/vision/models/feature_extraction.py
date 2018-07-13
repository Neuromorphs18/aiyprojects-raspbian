# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""API for Image Classification tasks."""

from src.aiy.vision.inference import ModelDescriptor
from src.aiy.vision.models import utils

# There are two models in our repository that can do image classification. One
# based on MobileNet model structure, the other based on SqueezeNet model
# structure.
#
# MobileNet based model has 59.9% top-1 accuracy on ImageNet.
# SqueezeNet based model has 45.3% top-1 accuracy on ImageNet.
MOBILENET = 'image_classification_mobilenet'
SQUEEZENET = 'image_classification_squeezenet'
_COMPUTE_GRAPH_NAME_MAP = {
    MOBILENET: 'mobilenet_v1_128res_0.5_features.binaryproto',
    SQUEEZENET: 'squeezenet_160res_5x5_0.75.binaryproto',
}
_OUTPUT_TENSOR_NAME_MAP = {
    MOBILENET: 'MobilenetV1/Predictions/Softmax',
    SQUEEZENET: 'Prediction',
}
_FEATURE_TENSOR_NAME_MAP = {
    MOBILENET: 'MobilenetV1/Predictions/Reshape',
}


def model(model_type=MOBILENET):
    return ModelDescriptor(
        name=model_type,
        input_shape=(1, 128, 128, 3),
        input_normalizer=(128.0, 128.0),
        compute_graph=utils.load_compute_graph(
            _COMPUTE_GRAPH_NAME_MAP[model_type]))


def get_output_features(result):
    """Get the final layer of the feature extractor."""
    assert len(result.tensors) == 1
    tensor_name = _FEATURE_TENSOR_NAME_MAP[result.model_name]
    tensor = result.tensors[tensor_name]
    features, shape = tensor.data, tensor.shape
    assert (shape.batch, shape.height, shape.width, shape.depth) == (1, 1, 1, 1001)

    return features
