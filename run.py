import cv2
import time
import argparse
import subprocess
import threading
import numpy as np
import sqlite3  # For database operations
from collections import deque
from datetime import datetime, timedelta
from model import Model

# Constants
FRAME_SKIP = 5  # Process every 5th frame
BUFFER_DURATION = timedelta(hours=1)  # Store frames for 1 hour
FRAME_WIDTH = 1280  # Default frame width
FRAME_HEIGHT = 720  # Default frame height
API_ENDPOINT = ""  # Leave blank to disable API functionality

def argument_parser():
    parser = argparse.ArgumentParser(description="Real-Time Violence Detection on RTSP Streams")
    parser.add_argument('--video-source', type=str, required=True, help='RTSP stream URL')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save detected frames')
    parser.add_argument('--api-endpoint', type=str, default='', help='API endpoint for sending notifications (optional)')
    return parser.parse_args()

def start_ffmpeg_stream(rtsp_url):
    """Launch FFmpeg to fetch the RTSP stream and output raw video frames."""
    return subprocess.Popen(
        [
            'ffmpeg', '-rtsp_transport', 'tcp', '-i', rtsp_url, '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-'
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
    )

def read_frame(ffmpeg_process):
    """Read a single frame from FFmpeg's output."""
    frame_size = FRAME_WIDTH * FRAME_HEIGHT * 3  # 3 channels (BGR)
    raw_frame = ffmpeg_process.stdout.read(frame_size)
    if len(raw_frame) != frame_size:
        print(f"Error: Expected {frame_size} bytes, got {len(raw_frame)} bytes.")
        return None
    return np.frombuffer(raw_frame, np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))

def save_to_database(event_type, timestamp, frame_path, confidence):
    """Save detected event to the SQLite database."""
    conn = sqlite3.connect('violence_events.db')
    c = conn.cursor()
    c.execute('''INSERT INTO events (event_type, timestamp, frame_path, confidence)
                 VALUES (?, ?, ?, ?)''',
              (event_type, timestamp, frame_path, confidence))
    conn.commit()
    conn.close()

def send_api_notification(timestamp, frame_path, confidence):
    """Send a detected event to the API endpoint (if provided)."""
    if not API_ENDPOINT:
        return  # Skip if no API endpoint is provided

    payload = {
        "event_type": "violence_detected",
        "timestamp": timestamp.isoformat(),
        "frame_path": frame_path,
        "confidence": confidence
    }
    try:
        response = requests.post(API_ENDPOINT, json=payload, timeout=5)  # Send POST request
        if response.status_code == 200:
            print("Notification sent successfully.")
        else:
            print(f"Failed to send notification. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending notification: {e}")

def process_frame(frame, model, frame_buffer, output_dir):
    """Process a single frame and store it if violence is detected."""
    result = model.predict(image=frame)
    label = result.get('label', '')
    confidence = result.get('confidence', 0.0)  # Assuming the model returns a confidence score

    if "violence" in label.lower():
        print(f"\U0001F6A8 Violence detected: {label}")
        timestamp = datetime.now()  # Store as datetime object
        frame_path = f"{output_dir}/frame_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(frame_path, frame)  # Save the frame as .jpg
        frame_buffer.append((timestamp, frame_path))  # Add to buffer
        cv2.imshow("⚠ Violence Detected!", frame)

        # Save event to the database
        save_to_database("violence_detected", timestamp.isoformat(), frame_path, confidence)

        # Send API notification (if endpoint is provided)
        send_api_notification(timestamp, frame_path, confidence)
    else:
        print("No violence detected. Skipping frame.")

def cleanup_buffer(frame_buffer):
    """Remove frames older than 1 hour from the buffer."""
    now = datetime.now()
    while frame_buffer and now - frame_buffer[0][0] > BUFFER_DURATION:
        _, old_frame_path = frame_buffer.popleft()
        try:
            os.remove(old_frame_path)  # Delete the old frame file
        except Exception as e:
            print(f"Error deleting file {old_frame_path}: {e}")

def frame_reader(ffmpeg_process, frame_queue):
    """Read frames from FFmpeg and add them to the queue."""
    while True:
        frame = read_frame(ffmpeg_process)
        if frame is None:
            print("Error: Could not read frame from FFmpeg stream. Reconnecting...")
            ffmpeg_process.kill()
            time.sleep(5)  # Wait before reconnecting
            ffmpeg_process = start_ffmpeg_stream(args.video_source)
            continue
        frame_queue.put(frame)

if __name__ == '__main__':
    args = argument_parser()
    model = Model()

    # Set API endpoint (if provided)
    API_ENDPOINT = args.api_endpoint

    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize FFmpeg process
    ffmpeg_process = start_ffmpeg_stream(args.video_source)

    # Frame buffer to store detected frames
    frame_buffer = deque()

    # Queue for frames to be processed
    from queue import Queue
    frame_queue = Queue(maxsize=10)  # Limit queue size to avoid memory issues

    # Start frame reader thread
    reader_thread = threading.Thread(target=frame_reader, args=(ffmpeg_process, frame_queue), daemon=True)
    reader_thread.start()

    frame_count = 0

    try:
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()

                # Skip frames to improve real-time performance
                frame_count += 1
                if frame_count % FRAME_SKIP != 0:
                    continue

                # Resize frame for faster processing
                frame = cv2.resize(frame, (640, 360))  # Adjust resolution as needed

                # Process frame
                process_frame(frame, model, frame_buffer, args.output_dir)

                # Clean up old frames from the buffer
                cleanup_buffer(frame_buffer)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Exiting...")

    # Cleanup
    ffmpeg_process.kill()
    cv2.destroyAllWindows()
    print("Video processing completed.")