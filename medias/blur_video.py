#\!/usr/bin/env python3
import cv2
import sys
import os

CASCADE_PATH = "/opt/homebrew/opt/opencv/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def pixelate_region(img, x, y, w, h, pixel_size=12):
    roi = img[y:y+h, x:x+w]
    small = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y:y+h, x:x+w] = pixelated
    return img

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video: {input_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    faces_found = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 3rd frame for face detection (speed optimization)
        # But apply blur based on last known face positions
        if frame_count % 3 == 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                faces_found = len(faces)
                last_faces = faces
        
        # Apply blur to detected faces
        if 'last_faces' in dir() and len(last_faces) > 0:
            for (x, y, w, h) in last_faces:
                padding = int(w * 0.25)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(width - x, w + 2*padding)
                h = min(height - y, h + 2*padding)
                frame = pixelate_region(frame, x, y, w, h, pixel_size=10)
        
        out.write(frame)
        
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count}/{total_frames} ({int(frame_count/total_frames*100)}%)", end='\r')
    
    cap.release()
    out.release()
    print(f"  Processed {frame_count} frames, found faces in video")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: blur_video.py <input> <output>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Processing: {os.path.basename(input_path)}")
    process_video(input_path, output_path)
    print(f"Saved to: {output_path}")
