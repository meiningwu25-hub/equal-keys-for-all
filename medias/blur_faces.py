#\!/usr/bin/env python3
import cv2
import os
import sys

# Paths
CASCADE_PATH = "/opt/homebrew/opt/opencv/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def pixelate_region(img, x, y, w, h, pixel_size=15):
    """Apply mosaic/pixelation to a region"""
    roi = img[y:y+h, x:x+w]
    # Resize down then up to create pixelation effect
    small = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y:y+h, x:x+w] = pixelated
    return img

def blur_faces_in_image(input_path, output_path):
    """Detect and blur faces in an image"""
    img = cv2.imread(input_path)
    if img is None:
        print(f"  Could not read: {input_path}")
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with multiple scale factors for better detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )
    
    print(f"  Found {len(faces)} face(s)")
    
    # Apply pixelation to each face
    for (x, y, w, h) in faces:
        # Expand region slightly to cover more of the face
        padding = int(w * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2*padding)
        h = min(img.shape[0] - y, h + 2*padding)
        img = pixelate_region(img, x, y, w, h, pixel_size=12)
    
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True

if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "blurred"
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"Processing: {filename}")
            blur_faces_in_image(input_path, output_path)
    
    print("\nDone\! Blurred images saved to:", output_dir)
