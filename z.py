import pyautogui
import time

# Get the screen size
screen_width, screen_height = pyautogui.size()

# Define the total duration (1 hour = 3600 seconds)
total_duration = 3600  # in seconds

# Define the duration for each movement (10 seconds)
movement_duration = 10  # in seconds

# Calculate the number of iterations needed
iterations = total_duration // movement_duration

# Loop for the specified number of iterations
for _ in range(iterations):
    # Move the mouse in a square pattern
    pyautogui.moveTo(100, 100, duration=0.25)  # move to (100, 100) over 0.25 seconds
    pyautogui.moveTo(screen_width - 100, 100, duration=0.25)  # move to (screen_width - 100, 100)
    pyautogui.moveTo(screen_width - 100, screen_height - 100, duration=0.25)  # move to bottom right corner
    pyautogui.moveTo(100, screen_height - 100, duration=0.25)  # move to bottom left corner

    # Wait for 10 seconds before the next movement
    time.sleep(movement_duration)

# Move the mouse back to the center of the screen
pyautogui.moveTo(screen_width / 2, screen_height / 2, duration=0.25)
