import time
import subprocess
import os
import signal
import RPi.GPIO as GPIO
from aiy.vision.leds import Leds, RgbLeds

# SET THE COMMANDLINE OPTIONS (PATH, RESOLOLUTION, ...) APPROPRIATELY!
aiy_command = 'python3 /home/pi/Repositories/aiyprojects-raspbian/src/' \
    'examples/vision/record_aiy_video.py ' \
    '-d /home/pi/Data_pi/aiy_frames/ --resolution 640 480 --frame_rate 5 ' \
    '--sensor_mode 7 --save_format .h264'
dvs_command = 'caer-bin -c /home/pi/Repositories/aiyprojects-raspbian/src/' \
    'examples/vision/davis-record-file.xml -o /outFile/ ' \
    'directory string /home/pi/Data_pi/davis/ &'

GPIO.setmode(GPIO.BCM)
GPIO_switch = 23
GPIO.setup(GPIO_switch, GPIO.IN)

print("Ready.")
print("Press push-button to start a recording.")
print("Press push-button again to stop the recording.")
print("Repeat to record new video.")
print("Press Ctrl-C to quit program.")
print("")

led_green = RgbLeds(Leds(), Leds.rgb_on((0, 255, 0)))
led_red = RgbLeds(Leds(), Leds.rgb_on((255, 0, 0)))
led_green.__enter__()

try:
    run = 0
    p1 = p2 = None
    while True:
        if GPIO.input(GPIO_switch) == 0 and run == 0:
            print("Started recording at time {}.".format(
                time.strftime("%Y_%m_%d_%H_%M_%S")))
            p1 = subprocess.Popen(aiy_command, shell=True,
                                  preexec_fn=os.setsid)
            led_red.__enter__()
            p2 = subprocess.Popen(dvs_command, shell=True,
                                  preexec_fn=os.setsid)
            run = 1
            while GPIO.input(GPIO_switch) == 0:
                time.sleep(0.1)
        if GPIO.input(GPIO_switch) == 0 and run == 1:
            os.killpg(p1.pid, signal.SIGINT)
            os.killpg(p2.pid, signal.SIGTERM)
            run = 0
            led_green.__enter__()
            print("Stopped current recording.")
            while GPIO.input(GPIO_switch) == 0:
                time.sleep(0.1)
except KeyboardInterrupt:
    print("Quit program.")
    led_green._leds.update(Leds.rgb_off())
    GPIO.cleanup()
