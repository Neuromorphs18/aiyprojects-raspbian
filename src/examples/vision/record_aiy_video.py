import os
import time
import argparse
import picamera


def main():
    """Image classification camera inference example."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--save_dir',
        '-d',
        type=str,
        dest='save_dir',
        default=os.path.join('/', 'home', 'pi', 'Data_pi', 'aiy_frames'),
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
        required=True,
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

    parser.add_argument(
        '--save_format',
        '-s',
        type=str,
        dest='save_format',
        default='.h264',
        help='Sets the path where the features will be stored.')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print("Saving aiy frames to {}.".format(args.save_dir))

    with picamera.PiCamera() as camera:
        camera.sensor_mode = args.sensor_mode
        camera.resolution = args.resolution
        camera.framerate = args.frame_rate
        path = os.path.join(args.save_dir, 'aiyOut-' +
                            time.strftime("%Y_%m_%d_%H_%M_%S") +
                            args.save_format)
        # print("Starting AIY recording.")
        camera.start_recording(path)
        try:
            while True:
                pass
        except KeyboardInterrupt:
            camera.stop_recording()
            # print("Stopped AIY recording.")


if __name__ == '__main__':
    main()
