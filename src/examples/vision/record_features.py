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
import time
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
        dest='save_frequency',
        help='Sets the number of feature vectors which are bundled for '
             'saving.')

    parser.add_argument(
        '--save_dir',
        '-d',
        type=str,
        dest='save_dir',
        default=os.path.join('/', 'home', 'pi', 'Data_pi', 'features'),
        help='Sets the path where the features will be stored.')

    parser.add_argument(
        '--frame_rate',
        '-f',
        type=int,
        dest='frame_rate',
        default=30,
        help='Sets the frame rate.')

    parser.add_argument(
        '--resolution',
        '-r',
        type=int,
        nargs='+',
        dest='resolution',
        help='Sets the resolution.')

    parser.add_argument(
        '--sensor_mode',
        '-m',
        type=int,
        dest='sensor_mode',
        default=4,
        help='Sets the sensor mode. For details see '
             'https://picamera.readthedocs.io/en/release-1.13/fov.html'
             '#sensor-modes')

    args = parser.parse_args()

    if args.resolution is None:
        args.resolution = (1640, 1232)

    if args.save_frequency is None:
        args.save_frequency = args.frame_rate

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print("Saving features to {}.".format(args.save_dir))

    with PiCamera() as camera:
        camera.sensor_mode = args.sensor_mode

        camera.resolution = args.resolution

        camera.framerate = args.frame_rate

        ts = time.strftime("%Y_%m_%d_%H_%M_%S")
        path_frames = os.path.join(args.save_dir, ts + '.h264')
        path_features = os.path.join(args.save_dir, ts)
        os.makedirs(path_features)

        # Start the camera stream.
        camera.start_recording(path_frames)

        try:
            with CameraInference(feature_extraction.model()) as inference:
                feature_list = []
                for i, result in enumerate(inference.run()):
                    if i == args.num_frames:
                        break
                    feature_list.append(feature_extraction.get_output_features(
                        result))
                    if i % args.save_frequency == 0:
                        print("Saved {} features".format(i))
                        np.save(os.path.join(path_features, str(i)),
                                np.concatenate(feature_list))
                        feature_list = []
        except KeyboardInterrupt:
            camera.stop_recording()
            # print("Stopped recording.")


if __name__ == '__main__':
    main()
