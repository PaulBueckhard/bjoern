from gpiozero import Button
from signal import pause

button = Button(17)  # BCM pin 17
button.when_pressed = lambda: print("Button pressed!")
print("Ready. Press the button.")
pause()
