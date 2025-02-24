import cv2
import time
import numpy as np
import streamlit as st
from PIL import Image
import subprocess
import threading
import queue
from collections import deque
from datetime import datetime, timedelta
import heapq  # For priority queue


@st.cache_resource  # Cache the model for better performance
def get_predictor_model():
    from model import Model
    model = Model()
    return model


# Constants
FRAME_SKIP = 3  # Process every 3rd frame
FRAME_WIDTH = 1280  # Default frame width
FRAME_HEIGHT = 720  # Default frame height


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


def process_frame(frame, model, detections_queue):
    """Process a single frame and add detections to the priority queue."""
    result = model.predict(image=frame)
    label = result.get('label', '')
    confidence = result.get('confidence', 0.0)  # Assuming the model returns a confidence score

    if "violence" in label.lower():
        timestamp = datetime.now()  # Store as datetime object
        frame_path = f"./output/frame_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(frame_path, frame)  # Save the frame as .jpg

        # Add detection to the priority queue (sorted by confidence in ascending order)
        heapq.heappush(detections_queue, (confidence, label, frame_path, frame))

        st.warning(f"🚨 Violence detected: {label} (Confidence: {confidence:.2f})")
    else:
        st.success("✅ No violence detected.")


def cleanup_buffer(detections_queue):
    """Remove old detections from the queue."""
    now = datetime.now()
    while detections_queue and now - datetime.strptime(detections_queue[0][2].split('_')[1], '%Y%m%d_%H%M%S_%f') > timedelta(hours=1):
        heapq.heappop(detections_queue)


def frame_reader(ffmpeg_process, frame_queue):
    """Read frames from FFmpeg and add them to the queue."""
    while True:
        frame = read_frame(ffmpeg_process)
        if frame is None:
            print("Error: Could not read frame from FFmpeg stream. Reconnecting...")
            ffmpeg_process.kill()
            time.sleep(5)  # Wait before reconnecting
            ffmpeg_process = start_ffmpeg_stream(rtsp_url)
            continue
        if not frame_queue.full():
            frame_queue.put(frame)


# Streamlit App
header = st.container()
model = get_predictor_model()

with header:
    st.title('Fairlens Violence Detection App')
    st.text(
        'Classifying whether there is a fight on a street, fire, car crash, or if everything is okay.')

# Option to upload an image or provide an RTSP URL
option = st.radio("Choose input type:", ("Upload an Image", "Live Stream (RTSP URL)"))

if option == "Upload an Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read and preprocess the image
        image = Image.open(uploaded_file).convert('RGB')
        image = np.array(image)

        # Perform prediction
        prediction = model.predict(image=image)
        label_text = prediction['label'].title()
        confidence = prediction['confidence']

        # Display results
        st.write(f'Predicted label is: **{label_text}**')
        st.write(f'Confidence: **{confidence:.2f}**')
        st.write('Original Image')
        st.image(image, use_column_width=True)

else:  # Live Stream (RTSP URL)
    rtsp_url = st.text_input("Enter RTSP URL:")
    if rtsp_url:
        st.write("Monitoring live stream...")

        # Create output directory
        import os
        os.makedirs("./output", exist_ok=True)

        # Initialize FFmpeg process
        ffmpeg_process = start_ffmpeg_stream(rtsp_url)

        # Priority queue for detections (sorted by confidence in ascending order)
        detections_queue = []

        # Queue for frames to be processed
        frame_queue = queue.Queue(maxsize=10)  # Limit queue size to avoid memory issues

        # Start frame reader thread
        reader_thread = threading.Thread(target=frame_reader, args=(ffmpeg_process, frame_queue), daemon=True)
        reader_thread.start()

        stframe = st.empty()  # Placeholder for the video frame
        frame_count = 0

        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()

                # Skip frames to improve real-time performance
                frame_count += 1
                if frame_count % FRAME_SKIP != 0:
                    continue

                # Resize frame for faster processing
                frame = cv2.resize(frame, (640, 360))  # Adjust resolution as needed

                # Display the frame in the Streamlit app
                stframe.image(frame, channels="BGR", use_column_width=True)

                # Process frame
                process_frame(frame, model, detections_queue)

                # Clean up old detections from the queue
                cleanup_buffer(detections_queue)

                # Display detections at the top of the queue
                if detections_queue:
                    st.write("### Recent Detections")
                    for confidence, label, frame_path, detection_frame in sorted(detections_queue, reverse=True):
                        st.image(detection_frame, caption=f"{label} (Confidence: {confidence:.2f})", use_column_width=True)