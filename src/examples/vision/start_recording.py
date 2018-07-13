import RPi.GPIO as GPIO
import time
import subprocess
import os
import signal
from aiy.vision.leds import Leds, RgbLeds, PrivacyLed


GPIO.setmode(GPIO.BCM)
GPIO_switch = 23
GPIO.setup(GPIO_switch, GPIO.IN)

print("Ready.")
print("Press push-button to start a recording.")
print("Press push-button again to stop the recording.")
print("Repeat to record new video.")
print("Press Ctrl-C to quit program.")
print("")

led = RgbLeds(Leds(), Leds.rgb_on((0, 255, 0)))
led.__enter__()
try:
    run = 0
    p1 = p2 = None
    while True:
        if GPIO.input(GPIO_switch) == 0 and run == 0:
            command_str1 = 'python3 /home/pi/Repositories/aiyprojects-' \
                           'raspbian/src/examples/vision/record_aiy_video.py' \
                           '  --resolution 240 180'
            p1 = subprocess.Popen(command_str1, shell=True,
                                  preexec_fn=os.setsid)
            print("Started recording.")
            PrivacyLed(Leds()).__enter__()
            RgbLeds(Leds(), Leds.rgb_on((255, 0, 0))).__enter__()
            command_str2 = 'caer-bin -c /home/pi/Repositories/caer/docs/' \
                           'davis-record-file_custom.xml -o /outFile/ ' \
                           'directory string /home/pi/Data_pi/davis/ &'
            p2 = subprocess.Popen(command_str2, shell=True,
                                  preexec_fn=os.setsid)
            run = 1
            while GPIO.input(GPIO_switch) == 0:
                time.sleep(0.1)
        if GPIO.input(GPIO_switch) == 0 and run == 1:
            print("Stopped current recording.")
            run = 0
            led.__enter__()
            os.killpg(p1.pid, signal.SIGINT)
            os.killpg(p2.pid, signal.SIGTERM)
            while GPIO.input(GPIO_switch) == 0:
                time.sleep(0.1)
except KeyboardInterrupt:
    print("Quit program.")
    led._leds.update(Leds.rgb_off())
    GPIO.cleanup()
