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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break
    cv2.imshow("GoPro Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
