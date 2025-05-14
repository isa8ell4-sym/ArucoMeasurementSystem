import cv2
import time

def open_camera(index, retries=10, delay=1.0):
    for i in range(retries):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.any():
                print(f"Camera opened on index {index} (try {i + 1})")
                return cap
        print(f"Retry {i + 1}/{retries}...")
        cap.release()
        time.sleep(delay)
    return None

cap = open_camera(1)  # Try different indices if needed

if cap is None:
    print("Could not open GoPro camera.")
    exit()

# Set up capture

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and output file
output_filename = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or 'H264' if supported
out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

print(f"Recording... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    out.write(frame)
    cv2.imshow('Recording', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopping...")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
