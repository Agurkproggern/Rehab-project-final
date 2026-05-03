import cv2
import os
import sys
import threading
import tkinter as tk
from tkinter import Label, Button, Frame, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import importlib
import time
import numpy as np
import hashlib


def crop_face_from_image(frame, margin_ratio=0.2):
    """
    Detect and crop face from image to make the face fill more of the frame.
    Assumes face is centered in the image.
    
    Args:
        frame: Input image (BGR format from OpenCV)
        margin_ratio: Margin around face as ratio of face size (0.2 = 20% margin)
    
    Returns:
        Cropped image with face filling more of the frame, or original frame if no face detected
    """
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        maxSize=(500, 500)
    )
    
    # If no face detected, return original frame
    if len(faces) == 0:
        print("  [CROP] No face detected, using original frame", flush=True)
        return frame
    
    # Use the largest face detected
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    
    # Calculate center of face
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    # Expand crop box with margin around face
    margin_w = int(w * margin_ratio)
    margin_h = int(h * margin_ratio)
    
    crop_w = w + 2 * margin_w
    crop_h = h + 2 * margin_h
    
    # Calculate crop boundaries, centered on face
    left = max(0, face_center_x - crop_w // 2)
    top = max(0, face_center_y - crop_h // 2)
    right = min(frame.shape[1], left + crop_w)
    bottom = min(frame.shape[0], top + crop_h)
    
    # Adjust if crop is out of bounds
    if right - left < crop_w:
        if left == 0:
            right = min(frame.shape[1], left + crop_w)
        else:
            left = max(0, right - crop_w)
    
    if bottom - top < crop_h:
        if top == 0:
            bottom = min(frame.shape[0], top + crop_h)
        else:
            top = max(0, bottom - crop_h)
    
    # Crop the image
    cropped_frame = frame[top:bottom, left:right]
    
    print(f"  [CROP] Face detected at ({x}, {y}) size {w}x{h}", flush=True)
    print(f"  [CROP] Cropped to {cropped_frame.shape[1]}x{cropped_frame.shape[0]}", flush=True)
    
    return cropped_frame


class WebcamCapture:
    """GUI application for webcam capture and head tilt prediction with manual button control."""
    
    def __init__(self, window_title="Webcam Head Tilt Prediction"):
        self.window = tk.Tk()
        self.window.title(window_title)
        self.window.geometry("800x800")
        
        # Thread control - initialize FIRST
        self.running = True
        self.capture_flag = False
        self.current_frame = None
        self.last_frame = None
        self.frame_lock = threading.Lock()
        self.frames_captured_count = 0
        self.frames_displayed_count = 0
        self.last_displayed_hash = None
        
        print("\n[WEBCAM GUI] Initializing Webcam Capture Application...")
        sys.stdout.flush()
        
        # Create GUI elements BEFORE webcam
        self.setup_gui()
        
        # Webcam setup - AFTER GUI is ready
        print("[WEBCAM GUI] Opening webcam...", flush=True)
        self.cap = cv2.VideoCapture(0)
        
        # Try alternative camera indices if 0 fails
        if not self.cap.isOpened():
            print("[WEBCAM GUI] Camera 0 failed, trying camera 1...", flush=True)
            self.cap = cv2.VideoCapture(1)
        
        if not self.cap.isOpened():
            print("[WEBCAM GUI] ERROR: Could not open any camera!", flush=True)
            messagebox.showerror("Error", "Could not open webcam. Please check:\n1. Camera is connected\n2. Camera is not in use\n3. Camera has proper permissions")
            self.running = False
            self.window.destroy()
            return
        
        print("[WEBCAM GUI] Camera opened successfully", flush=True)
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Save folder
        self.save_folder = "captured_images"
        os.makedirs(self.save_folder, exist_ok=True)
        
        # Start video feed thread
        self.video_thread = threading.Thread(target=self.capture_video_frames, daemon=True)
        self.video_thread.start()
        print("[WEBCAM GUI] Capture thread started", flush=True)
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Schedule first update - this will start the display loop on the main thread
        print("[WEBCAM GUI] Scheduling first display update", flush=True)
        self.window.after(100, self.update_display)
    
    
    def setup_gui(self):
        """Create GUI layout with video feed, buttons, and result display."""
        
        # Top frame for video feed
        video_frame = Frame(self.window, bg="black")
        video_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.video_label = Label(video_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Middle frame for prediction result
        result_frame = Frame(self.window, bg="lightgray", height=60)
        result_frame.pack(padx=10, pady=5, fill=tk.X)
        
        self.result_label = Label(
            result_frame,
            text="Ready. Click 'Capture' to predict.",
            font=("Arial", 12, "bold"),
            bg="lightgray",
            fg="blue",
            wraplength=600
        )
        self.result_label.pack(pady=10)
        
        # Bottom frame for buttons
        button_frame = Frame(self.window)
        button_frame.pack(padx=10, pady=10, fill=tk.X)
        
        # Capture button
        self.capture_btn = Button(
            button_frame,
            text="📸 CAPTURE & PREDICT",
            command=self.trigger_capture,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            width=20
        )
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Quit button
        quit_btn = Button(
            button_frame,
            text="❌ QUIT",
            command=self.on_closing,
            bg="#f44336",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            width=10
        )
        quit_btn.pack(side=tk.RIGHT, padx=5)
    
    def capture_video_frames(self):
        """Background thread: continuously capture frames from webcam."""
        print("[CAPTURE THREAD] Starting frame capture loop", flush=True)
        failed_reads = 0
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                failed_reads += 1
                print(f"[CAPTURE THREAD] Failed to read frame (attempt {failed_reads})", flush=True)
                if failed_reads > 10:
                    print("[CAPTURE THREAD] Too many failed reads, stopping", flush=True)
                    break
                time.sleep(0.1)
                continue
            
            failed_reads = 0
            self.frames_captured_count += 1
            
            # Store current frame for display
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Check if capture flag is set
            if self.capture_flag:
                with self.frame_lock:
                    self.last_frame = frame.copy()
                self.capture_flag = False
                print(f"[CAPTURE THREAD] Frame captured for prediction (frame #{self.frames_captured_count})", flush=True)
            
            # Small delay
            time.sleep(0.01)
    
    def update_display(self):
        """Update video display - called from main thread."""
        if not self.running:
            return
        
        try:
            with self.frame_lock:
                frame = self.current_frame.copy() if self.current_frame is not None else None
            
            if frame is not None:
                # Resize for display
                display_frame = cv2.resize(frame, (640, 480))
                
                # Convert for tkinter display
                image_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Convert numpy array to PIL Image
                image_pil = Image.fromarray(image_rgb)
                
                # Convert to PhotoImage
                image_tk = ImageTk.PhotoImage(image=image_pil)
                
                # Update label - this is where it actually displays
                self.video_label.config(image=image_tk)
                self.video_label.image = image_tk  # Keep reference
                
                self.frames_displayed_count += 1
                
                if self.frames_displayed_count % 30 == 0:  # Log every 30 frames
                    print(f"[DISPLAY] ✓ Displayed {self.frames_displayed_count} frames, Captured: {self.frames_captured_count}", flush=True)
            else:
                if self.frames_displayed_count == 0:
                    print(f"[DISPLAY] Waiting for frames... (captured: {self.frames_captured_count})", flush=True)
        except Exception as e:
            print(f"[VIDEO UPDATE ERROR] {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        
        # Schedule next update in 33ms (~30 FPS)
        if self.running:
            self.window.after(33, self.update_display)
    
    def trigger_capture(self):
        """Trigger capture on next frame."""
        print("\n[BUTTON CLICKED] Starting prediction process...", flush=True)
        print(f"[STATE] Frames captured so far: {self.frames_captured_count}", flush=True)
        
        self.result_label.config(
            text="Capturing image...",
            fg="orange"
        )
        self.capture_btn.config(state=tk.DISABLED)
        
        # Grab the current frame directly
        with self.frame_lock:
            if self.current_frame is None:
                print("[ERROR] No frame available yet", flush=True)
                self.result_label.config(text="✗ No frame available", fg="#f44336")
                self.capture_btn.config(state=tk.NORMAL)
                return
            
            frame_to_predict = self.current_frame.copy()
        
        print(f"[TRIGGER] Captured frame at {datetime.now()}", flush=True)
        
        # Give time for a fresh new frame to arrive from camera (~150ms for 3-4 frames)
        self.window.after(150, lambda: self._start_prediction(frame_to_predict))
    
    def _start_prediction(self, frame):
        """Start prediction in a separate thread."""
        print("[TIMER] Starting async prediction thread...", flush=True)
        
        print(f"[TIMER] Frame shape: {frame.shape}", flush=True)
        
        # Run prediction in a separate thread
        thread = threading.Thread(target=self._predict_async, args=(frame,), daemon=False)
        thread.start()
    
    def _predict_async(self, frame):
        """Async prediction worker (runs in background)."""
        print("\n[ASYNC START] Prediction thread started", flush=True)
        
        try:
            print("="*60, flush=True)
            print("[1] SAVING IMAGE", flush=True)
            print("="*60, flush=True)
            
            print(f"[1] Frame info - Shape: {frame.shape}, Type: {frame.dtype}, Min: {frame.min()}, Max: {frame.max()}", flush=True)
            
            # Create hash of input frame to verify different images
            frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:8]
            print(f"[1.0A] Input frame hash: {frame_hash}", flush=True)
            
            # Crop face from image to make it fill more of the frame
            print("[1.0] Detecting and cropping face...", flush=True)
            cropped_frame = crop_face_from_image(frame, margin_ratio=0.2)
            
            print(f"[1.0.5] Cropped frame info - Shape: {cropped_frame.shape}, Type: {cropped_frame.dtype}", flush=True)
            
            # Create hash of cropped frame
            cropped_hash = hashlib.md5(cropped_frame.tobytes()).hexdigest()[:8]
            print(f"[1.0B] Cropped frame hash: {cropped_hash}", flush=True)
            
            # Save image
            filename = datetime.now().strftime("img_%Y%m%d_%H%M%S.jpg")
            image_path = os.path.join(self.save_folder, filename)
            success = cv2.imwrite(image_path, cropped_frame)
            
            print(f"[1.1] Image saved to: {image_path} (success: {success})", flush=True)
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            size_bytes = os.path.getsize(image_path)
            print(f"[1.2] File size: {size_bytes} bytes", flush=True)
            
            print("\n" + "="*60, flush=True)
            print("[2] IMPORTING MODULES", flush=True)
            print("="*60, flush=True)
            
            # Import modules fresh and reload to avoid caching
            import sys
            if 'pipeline.pain_prediction' in sys.modules:
                del sys.modules['pipeline.pain_prediction']
            if 'pipeline.Resnet_rebuild' in sys.modules:
                del sys.modules['pipeline.Resnet_rebuild']
            
            from pipeline import pain_prediction as pp_module
            print("[2.1] pain_prediction module imported (fresh)", flush=True)
            
            # Get components
            model = pp_module.trained_model
            transform = pp_module.transform
            device = pp_module.device
            
            print(f"[2.2] Model type: {type(model)}", flush=True)
            print(f"[2.3] Transform type: {type(transform)}", flush=True)
            print(f"[2.4] Device: {device}", flush=True)
            
            print("\n" + "="*60, flush=True)
            print("[3] RUNNING PREDICTION", flush=True)
            print("="*60, flush=True)
            
            # Run prediction
            prediction = pp_module.predict_tilt(image_path, model, transform, device)
            
            print(f"\n[3.1] PREDICTION COMPLETE!", flush=True)
            print(f"[3.2] RESULT: {prediction}", flush=True)
            print("="*60 + "\n", flush=True)
            
            # Update GUI result
            result_text = f"✓ Prediction: {prediction}"
            self.window.after(0, lambda: self.update_result(result_text, "#4CAF50"))
            
        except Exception as e:
            print("\n" + "="*60, flush=True)
            print("[ERROR] PREDICTION FAILED", flush=True)
            print("="*60, flush=True)
            print(f"[E1] Error type: {type(e).__name__}", flush=True)
            print(f"[E2] Error message: {str(e)}", flush=True)
            
            import traceback
            print("\n[E3] Full traceback:", flush=True)
            traceback.print_exc(file=sys.stdout)
            print("="*60 + "\n", flush=True)
            sys.stdout.flush()
            
            error_text = f"✗ {type(e).__name__}: {str(e)[:40]}"
            self.window.after(0, lambda: self.update_result(error_text, "#f44336"))
    
    def update_result(self, text, color):
        """Update result label in GUI."""
        print(f"[UPDATE GUI] {text}", flush=True)
        self.result_label.config(text=text, fg=color)
        self.capture_btn.config(state=tk.NORMAL)
    
    def on_closing(self):
        """Close application cleanly."""
        print("[CLOSING] Shutting down webcam...", flush=True)
        self.running = False
        self.window.quit()
        self.cap.release()
        self.window.destroy()
        print("[CLOSED] Webcam closed", flush=True)
    
    def start(self):
        """Start the GUI application."""
        print("[MAINLOOP] Starting tkinter main loop...", flush=True)
        self.window.mainloop()


def capture_with_button_gui():
    """
    Launch a GUI application for webcam capture with manual button control.
    """
    print("\n" + "="*70)
    print("LAUNCHING WEBCAM GUI APPLICATION")
    print("="*70 + "\n", flush=True)
    
    app = WebcamCapture("Head Tilt Prediction - Click to Capture")
    app.start()


def delete_captured_images(folder="captured_images"):
    """
    Delete all captured images from the specified folder.
    
    Args:
        folder: Path to the folder containing images (default: "captured_images")
    """
    print("\n" + "="*70)
    print("DELETING CAPTURED IMAGES")
    print("="*70, flush=True)
    
    try:
        if not os.path.exists(folder):
            print(f"✓ Folder '{folder}' does not exist - nothing to delete", flush=True)
            return
        
        # Get all image files
        files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if not files:
            print(f"✓ No images found in '{folder}'", flush=True)
            return
        
        # Delete each file
        deleted_count = 0
        for filename in files:
            file_path = os.path.join(folder, filename)
            try:
                os.remove(file_path)
                deleted_count += 1
                print(f"  ✓ Deleted: {filename}", flush=True)
            except Exception as e:
                print(f"  ✗ Could not delete {filename}: {e}", flush=True)
        
        print(f"\n✓ Successfully deleted {deleted_count} image(s)", flush=True)
        print("="*70 + "\n", flush=True)
        
    except Exception as e:
        print(f"✗ Error deleting images: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    capture_with_button_gui()