import subprocess
from gpiozero import Button
from signal import pause

button = Button(17)
recording_process = None  # will hold arecord process

def start_recording():
    global recording_process
    print("Recording... (hold button)")
    # Start recording until killed
    recording_process = subprocess.Popen([
        "arecord", "--format=cd", "input.wav"
    ])

def stop_recording():
    global recording_process
    if recording_process:
        print("Stopping recording...")
        recording_process.terminate()
        recording_process.wait()
        recording_process = None
        print("Playing back...")
        subprocess.run(["aplay", "input.wav"])
        print("Done. Press and hold again to record.")

button.when_pressed = start_recording
button.when_released = stop_recording

print("Hold the button to record, release to stop and playback.")
pause()
