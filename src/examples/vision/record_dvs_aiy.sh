#! /bin/bash

echo Running script to record data from DAVIS and AIY Vision Kit.

echo Starting DAVIS recording.

caer-bin -c /home/pi/Repositories/caer/docs/davis-record-file_custom.xml -o /outFile/ directory string /home/pi/Data_pi/davis/ &

python3 /home/pi/Repositories/aiyprojects-raspbian/src/examples/vision/record_video.py --resolution 240 180

echo Finished recording.
