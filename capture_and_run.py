import cv2
import time
import subprocess
import firebase_admin
from firebase_admin import credentials, storage, db
import geocoder
import os
from pathlib import Path

# Initialize Firebase Admin SDK
cred = credentials.Certificate('credentials.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'wildcare-263bc.appspot.com',
    'databaseURL': 'https://wildcare-263bc-default-rtdb.firebaseio.com/'
})

# Capture image from camera
def capture_image(image_path):
    cap = cv2.VideoCapture(0)  # Open the default camera
    
    # Set camera resolution (e.g., 1920x1080 for full HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return False
    
    # Give the camera some time to warm up
    time.sleep(1)  # 1 second delay for camera initialization
    
    # Capture a few frames to ensure the camera is stable
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            # Optionally display the frame for a brief moment (can be commented out)
            cv2.imshow('Capture', frame)
            cv2.waitKey(100)  # Display the frame for 100 ms
        else:
            print("Can't receive frame (stream end?). Exiting ...")
            cap.release()
            cv2.destroyAllWindows()
            return False
    
    # Capture the final frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    # Save the captured image
    cv2.imwrite(image_path, frame)
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    return True

# Run YOLOv9 detection
def run_detection(image_path, weights='yolov9-c.pt'):
    # Define the command to run YOLO detection
    command = f'python detect.py --weights {weights} --source {image_path} --device cpu'
    
    # Run the YOLO detection script
    subprocess.run(command, shell=True)
    
    # Find the most recent output directory
    output_dir = Path('runs/detect')
    latest_exp_dir = max(output_dir.glob('exp*'), key=os.path.getmtime)
    
    # Define the path to the YOLO output image
    result_image_path = latest_exp_dir / 'captured_image.jpg'
    
    # Check if the YOLO result image was saved successfully
    if result_image_path.exists():
        return str(result_image_path)
    else:
        print("YOLO detection output image not found.")
        return None

# Upload image to Firebase Storage
def upload_image_to_storage(image_path, upload_path):
    bucket = storage.bucket()
    blob = bucket.blob(upload_path)
    blob.upload_from_filename(image_path)
    blob.make_public()  # Make the file publicly accessible
    return blob.public_url

# Get current coordinates
def get_coordinates():
    g = geocoder.ip('me')
    if g.ok:
        return {
            'latitude': g.latlng[0],
            'longitude': g.latlng[1]
        }
    else:
        return {
            'latitude': 0.0,
            'longitude': 0.0
        }

# Store image URL and coordinates in Firebase Realtime Database
def store_image_and_coordinates_in_db(original_image_url, detection_image_url, coordinates):
    ref = db.reference('images')
    ref.push({
        'original_url': original_image_url,
        'detection_url': detection_image_url,
        'coordinates': coordinates
    })

if __name__ == '__main__':
    try:
        while True:
            image_path = 'captured_image.jpg'
            if capture_image(image_path):
                # Run YOLO detection and get the result image path
                result_image_path = run_detection(image_path)

                if result_image_path:
                    # Upload images to Firebase Storage
                    original_image_url = upload_image_to_storage(image_path, 'images/original_image.jpg')
                    detection_image_url = upload_image_to_storage(result_image_path, 'images/detection_image.jpg')
                    
                    # Get coordinates
                    coordinates = get_coordinates()
                    
                    # Store URLs and coordinates in Firebase Realtime Database
                    store_image_and_coordinates_in_db(original_image_url, detection_image_url, coordinates)
                    print('Image URLs and coordinates stored in Realtime Database successfully.')
                else:
                    print("YOLO detection failed. Detection image not uploaded.")
            else:
                print("Image capture failed. Detection not run.")
            
            # Wait until current iteration is done before starting the next
            time.sleep(0.5)  # Adjust the sleep time as needed

    except KeyboardInterrupt:
        print("Program terminated by user.")
