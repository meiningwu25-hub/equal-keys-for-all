#\!/usr/bin/env python3
import cv2
import os
import sys

CASCADE_PATH = "/opt/homebrew/opt/opencv/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def pixelate_region(img, x, y, w, h, pixel_size=12):
    """Apply mosaic/pixelation to a region"""
    roi = img[y:y+h, x:x+w]
    small = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y:y+h, x:x+w] = pixelated
    return img

def blur_faces_in_image(input_path, output_path):
    """Detect faces and blur only smaller ones (kids), keep largest (Meining)"""
    img = cv2.imread(input_path)
    if img is None:
        print(f"  Could not read: {input_path}")
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        print(f"  Found 0 faces")
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    
    # Find the largest face (assumed to be Meining - the older teenager)
    face_areas = [(x, y, w, h, w*h) for (x, y, w, h) in faces]
    face_areas.sort(key=lambda f: f[4], reverse=True)
    largest_area = face_areas[0][4]
    
    # Blur faces that are significantly smaller than the largest (likely younger kids)
    # Threshold: blur if face area is less than 70% of the largest face
    threshold = largest_area * 0.70
    
    blurred_count = 0
    kept_count = 0
    
    for (x, y, w, h, area) in face_areas:
        if area < threshold:
            # This is likely a younger kid - blur it
            padding = int(w * 0.2)
            bx = max(0, x - padding)
            by = max(0, y - padding)
            bw = min(img.shape[1] - bx, w + 2*padding)
            bh = min(img.shape[0] - by, h + 2*padding)
            img = pixelate_region(img, bx, by, bw, bh, pixel_size=12)
            blurred_count += 1
        else:
            # This is likely Meining (older/larger face) - keep it
            kept_count += 1
    
    print(f"  Found {len(faces)} face(s): blurred {blurred_count}, kept {kept_count} (Meining)")
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
    
    print("\nDone\!")
