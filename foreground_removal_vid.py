import cv2
import numpy as np

# Path to the input video file
video_path = 'video.mp4'

# Initialize the video capture object
cap = cv2.VideoCapture(video_path)

# Check if the video capture has been initialized correctly
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Create the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Read the first frame to capture the background
ret, background = cap.read()

if not ret:
    print("Error: Could not read the first frame.")
    exit()

# Flip the background frame
background = np.flip(background, axis=1)

# Process each frame
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

    # Create an image showing static background frame pixels only for the masked region
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
