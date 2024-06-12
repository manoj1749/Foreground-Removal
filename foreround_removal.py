import cv2
import time
import numpy as np

# Initialize the video capture object
cap = cv2.VideoCapture(1)

# Allow the system to sleep for 3 seconds before the webcam starts
time.sleep(3)
count = 0
background = None

# Create the background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

# Capture the background frame for a few seconds
for i in range(60):
    ret, frame = cap.read()
    if not ret:
        continue
    frame = np.flip(frame, axis=1)
    # Apply the background subtractor to get the initial background
    fgbg.apply(frame)
    if background is None:
        background = frame

# Detect the defined color portion in each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)

    # Apply the background subtractor to get the foreground mask
    fgmask = fgbg.apply(frame)

    # Remove shadows (if any)
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    # Apply some morphological operations to remove noise and fill in the holes
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

    # Create an inverted mask to segment out the moving object from the frame
    mask_inv = cv2.bitwise_not(fgmask)

    # Segment the moving object out of the frame using bitwise_and with the inverted mask
    res1 = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Create image showing static background frame pixels only for the masked region
    res2 = cv2.bitwise_and(background, background, mask=fgmask)

    # Generate the final output by adding res1 and res2
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Display the output to the screen
    cv2.imshow("Foreground Removal", final_output)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
