#!/usr/bin/env python3
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
"""Camera image classification demo code.

Runs continuous image detection on the VisionBonnet and prints the object and
probability for top three objects.

Example:
image_classification_camera.py --num_frames 10
"""
import argparse

import os
import numpy as np
from src.aiy.vision.inference import CameraInference
from src.aiy.vision.models import feature_extraction
from picamera import PiCamera


def main():
    """Image classification camera inference example."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_frames',
        '-n',
        type=int,
        dest='num_frames',
        default=-1,
        help='Sets the number of frames to run for, otherwise runs forever.')

    parser.add_argument(
        '--num_objects',
        '-c',
        type=int,
        dest='num_objects',
        default=3,
        help='Sets the number of object interences to print.')

    parser.add_argument(
        '--save_frequency',
        '-s',
        type=int,
        dest='fps',
        help='Sets the number of feature vectors which are bundled for '
             'saving.')

    parser.add_argument(
        '--save_dir',
        '-d',
        type=str,
        dest='save_dir',
        default=os.path.join('/', 'home', 'pi', 'Data', 'features'),
        help='Sets the path where the features will be stored.')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print("Saving features to {}.".format(args.save_dir))
    if len(os.listdir(args.save_dir)) != 0:
        print("WARNING: Directory not empty.")

    with PiCamera() as camera:
        # Forced sensor mode, 1640x1232, full FoV. See:
        # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
        # This is the resolution inference run on.
        camera.sensor_mode = 4

        # Scaled and cropped resolution. If different from sensor mode implied
        # resolution, inference results must be adjusted accordingly. This is
        # true in particular when camera.start_recording is used to record an
        # encoded h264 video stream as the Pi encoder can't encode all native
        # sensor resolutions, or a standard one like 1080p may be desired.
        camera.resolution = (1640, 1232)

        # Start the camera stream.
        camera.framerate = 30
        camera.start_preview()

        if args.fps is None:
            args.fps = camera.framerate

        with CameraInference(feature_extraction.model()) as inference:
            feature_list = []
            for i, result in enumerate(inference.run()):
                if i == args.num_frames:
                    break
                feature_list.append(feature_extraction.get_output_features(
                    result))
                if i % args.fps == 0:
                    print("Saved {} features".format(i))
                    np.save(os.path.join(args.save_dir, str(i)),
                            np.concatenate(feature_list))
                    feature_list = []

        camera.stop_preview()


if __name__ == '__main__':
    main()
